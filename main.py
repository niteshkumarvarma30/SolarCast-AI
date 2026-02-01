import os
import joblib
import pandas as pd
import numpy as np
import pvlib
import requests
import xgboost as xgb
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv() 
app = FastAPI(title="SolarCast Enterprise API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI ASSET SYNC ---
expected_columns = ["Month", "Hour", "Temperature", "Relative Humidity", "Pressure", "Wind Speed", "Solar Zenith Angle", "Clearsky GHI", "lat", "long", "Hour_sin", "Hour_cos", "Month_sin", "Month_cos"]
model, pt_y = None, None

try:
    model_obj = joblib.load('assets/solar_xgboost_model_v2.pkl')
    model = model_obj.get_booster() if hasattr(model_obj, "get_booster") else model_obj
    pt_y = joblib.load('assets/target_transformer_v2.pkl')
    model.feature_names = expected_columns
    print("✅ Enterprise Assets Loaded.")
except Exception as e:
    print(f"❌ Initialization Error: {e}")

class ForecastRequest(BaseModel):
    pincode: str = Field(..., description="Target Location ZIP")
    unit_cost: float = Field(..., description="Current Utility Rate")

def get_location_data(pincode: str):
    backups = {"821304": (24.91, 84.18, "Dehri"), "110001": (28.61, 77.23, "New Delhi")}
    if pincode in backups: return backups[pincode]
    try:
        url = f"https://nominatim.openstreetmap.org/search?postalcode={pincode}&country=India&format=json&limit=1"
        res = requests.get(url, headers={'User-Agent': 'SolarCast/3.0'}).json()
        if res:
            city = res[0].get('display_name', '').split(',')[0]
            return float(res[0]['lat']), float(res[0]['lon']), city
    except: pass
    return None, None, None

@app.post("/predict")
async def get_live_inference(req: ForecastRequest):
    try:
        lat, lon, city = get_location_data(req.pincode)
        if lat is None: raise HTTPException(status_code=400, detail="Invalid Geo-Location")

        API_KEY = os.getenv("TOMORROW_API_KEY")
        weather_url = f"https://api.tomorrow.io/v4/weather/realtime?location={lat},{lon}&apikey={API_KEY}"
        w_res = requests.get(weather_url, timeout=5).json()
        
        condition = "Live AI Mode Active"
        if 'data' not in w_res:
            vals = {'temperature': 27, 'humidity': 45, 'pressureSurfaceLevel': 1012, 'windSpeed': 4}
            condition = "API LIMIT: PHYSICS FALLBACK ACTIVE"
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
        prediction = max(0, float(pred_raw))

        return {
            "prediction": round(prediction, 2), "city": city, "lat": lat, "lon": lon,
            "revenue_hourly": round((prediction / 1000) * req.unit_cost, 2),
            "status": condition
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/outlook")
async def get_enterprise_outlook(lat: float, lon: float, cost: float):
    location = pvlib.location.Location(lat, lon)
    start_date = pd.Timestamp.now(tz='UTC').normalize()
    daily_results = []
    
    capacities = [1.0, 3.0, 5.0]
    totals = {cap: 0 for cap in capacities}

    for i in range(15):
        day = start_date + pd.Timedelta(days=i)
        samples = [9, 12, 15]
        flux_values = []
        for hr in samples:
            ts = day + pd.Timedelta(hours=hr)
            clearsky = location.get_clearsky(pd.DatetimeIndex([ts]))
            flux_values.append(clearsky['ghi'].iloc[0])
        
        avg_flux = np.mean(flux_values)
        for cap in capacities:
            # ROI Math: (Flux/1000) * kW * 5 peak hours
            totals[cap] += (avg_flux / 1000) * cap * 5
            
        daily_results.append({"day": str(day.date()), "money": round((avg_flux/1000)*1*5*cost, 2)})

    return {
        "outlook": daily_results,
        "comparison": [
            {"size": f"{cap}kW", "units": round(totals[cap], 1), "savings": round(totals[cap]*cost, 0)}
            for cap in capacities
        ]
    }