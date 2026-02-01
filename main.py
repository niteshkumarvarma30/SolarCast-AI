import os
import joblib
import pandas as pd
import numpy as np
import pvlib
import requests
import xgboost as xgb
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv() 
app = FastAPI(title="SolarCast AI - 2026 Profit Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ASSET LOADING ---
expected_columns = ["Month", "Hour", "Temperature", "Relative Humidity", "Pressure", "Wind Speed", "Solar Zenith Angle", "Clearsky GHI", "lat", "long", "Hour_sin", "Hour_cos", "Month_sin", "Month_cos"]
model, pt_y = None, None

try:
    model_obj = joblib.load('assets/solar_xgboost_model_v2.pkl')
    model = model_obj.get_booster() if hasattr(model_obj, "get_booster") else model_obj
    pt_y = joblib.load('assets/target_transformer_v2.pkl')
    model.feature_names = expected_columns
    print("✅ Assets Synchronized.")
except Exception as e:
    print(f"❌ Asset Error: {e}")

class SolarFinancialRequest(BaseModel):
    pincode: str
    avg_daily_usage: float # Fixed at 10 units/day for comparison
    unit_cost: float       # Fixed at ₹6/unit for comparison

def get_coords_with_city(pincode: str):
    backups = {"812001": (25.24, 86.97, "Bhagalpur"), "821304": (24.91, 84.18, "Dehri"), "110001": (28.61, 77.23, "New Delhi")}
    if pincode in backups: return backups[pincode]
    try:
        url = f"https://nominatim.openstreetmap.org/search?postalcode={pincode}&country=India&format=json&limit=1"
        res = requests.get(url, headers={'User-Agent': 'SolarCast/2.0'}).json()
        if res:
            city = res[0].get('display_name', '').split(',')[0]
            return float(res[0]['lat']), float(res[0]['lon']), city
    except: pass
    return None, None, None

@app.post("/predict")
async def predict_solar(req: SolarFinancialRequest):
    try:
        lat, lon, city = get_coords_with_city(req.pincode)
        if lat is None: raise HTTPException(status_code=400, detail="Invalid Pincode")

        API_KEY = os.getenv("TOMORROW_API_KEY")
        weather_url = f"https://api.tomorrow.io/v4/weather/realtime?location={lat},{lon}&apikey={API_KEY}"
        w_res = requests.get(weather_url, timeout=5).json()
        
        condition = "Analysis Successful"
        if 'data' not in w_res:
            vals = {'temperature': 25, 'humidity': 50, 'pressureSurfaceLevel': 1013, 'windSpeed': 5}
            condition = "API LIMIT REACHED (Using Physics Fallback)"
        else:
            vals = w_res['data']['values']
        
        ts = pd.Timestamp.now(tz='UTC')
        loc = pvlib.location.Location(lat, lon)
        solpos = loc.get_solarposition(pd.DatetimeIndex([ts]))
        clearsky = loc.get_clearsky(pd.DatetimeIndex([ts]))

        input_df = pd.DataFrame([{
            "Month": ts.month, "Hour": ts.hour, "Temperature": vals.get('temperature'),
            "Relative Humidity": vals.get('humidity'), "Pressure": vals.get('pressureSurfaceLevel'),
            "Wind Speed": vals.get('windSpeed'), "Solar Zenith Angle": solpos['zenith'].iloc[0],
            "Clearsky GHI": clearsky['ghi'].iloc[0], "lat": lat, "long": lon,
            "Hour_sin": np.sin(2*np.pi*ts.hour/24), "Hour_cos": np.cos(2*np.pi*ts.hour/24),
            "Month_sin": np.sin(2*np.pi*ts.month/12), "Month_cos": np.cos(2*np.pi*ts.month/12)
        }])
        
        dmatrix = xgb.DMatrix(input_df[expected_columns], feature_names=expected_columns)
        pred_raw = pt_y.inverse_transform(model.predict(dmatrix).reshape(-1, 1))[0][0]
        prediction = max(0, round(float(pred_raw), 2))

        return {
            "prediction": prediction, "city": city, "lat": lat, "lon": lon,
            "savings_hourly": round((prediction / 1000) * req.unit_cost, 2),
            "metadata": {"temp": vals.get('temperature'), "condition": condition}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/outlook")
async def get_outlook(lat: float, lon: float, kw: float = 1.0):
    location = pvlib.location.Location(lat, lon)
    start_date = pd.Timestamp.now(tz='UTC').normalize()
    results = []
    
    # Grid Baseline: 15 days * 10 units * ₹6 = ₹900
    grid_baseline = 900 
    total_solar_revenue = 0

    for i in range(15):
        day = start_date + pd.Timedelta(days=i)
        # Average Daylight Samples (9 AM, 12 PM, 3 PM)
        samples = [9, 12, 15]
        flux_values = []
        for hr in samples:
            ts = day + pd.Timedelta(hours=hr)
            clearsky = location.get_clearsky(pd.DatetimeIndex([ts]))
            flux_values.append(clearsky['ghi'].iloc[0])
        
        avg_flux = np.mean(flux_values)
        # Daily Solar Units = (Avg Flux / 1000) * 1kW * 5 peak hours
        daily_units = (avg_flux / 1000) * kw * 5
        daily_savings = daily_units * 6 # Fixed ₹6 rate
        total_solar_revenue += daily_savings
        
        results.append({"day": str(day.date()), "avg_flux": round(avg_flux, 2), "daily_profit": round(daily_savings, 2)})

    return {
        "outlook": results,
        "grid_baseline": grid_baseline,
        "solar_total": round(total_solar_revenue, 2),
        "net_profit": round(total_solar_revenue - grid_baseline, 2)
    }