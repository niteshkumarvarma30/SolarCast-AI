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
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv() 
app = FastAPI(title="SolarCast Enterprise API")

# --- GLOBAL ACCESS CONFIG ---
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
    print("✅ Assets Synchronized.")
except Exception as e:
    print(f"❌ Asset Error: {e}")

class ForecastRequest(BaseModel):
    pincode: str = Field(..., description="Target Location PIN")
    unit_cost: float = Field(..., description="Utility Rate")

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
        if lat is None: raise HTTPException(status_code=400, detail="Invalid PIN")
        API_KEY = os.getenv("TOMORROW_API_KEY")
        weather_url = f"https://api.tomorrow.io/v4/weather/realtime?location={lat},{lon}&apikey={API_KEY}"
        w_res = requests.get(weather_url, timeout=5).json()
        
        condition = "Live AI Mode Active"
        vals = w_res.get('data', {}).get('values', {'temperature': 27, 'humidity': 45, 'pressureSurfaceLevel': 1012, 'windSpeed': 4})
        if 'data' not in w_res: condition = "API LIMIT: PHYSICS FALLBACK ACTIVE"
        
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
            totals[cap] += (avg_flux / 1000) * cap * 5
        daily_results.append({"day": str(day.date()), "money": round((avg_flux/1000)*1*5*cost, 2)})

    hourly_results = []
    for i in range(24):
        hour_ts = start_date + pd.Timedelta(hours=i)
        mock_flux = max(0, 800 * np.sin(np.pi * (i - 6) / 12)) if 6 <= i <= 18 else 0
        hourly_results.append({"hour": hour_ts.strftime("%H:00"), "flux": round(mock_flux, 2)})

    # System Cost Estimations
    cost_data = {
        1.0: {"pre": 45000, "subsidy": 30000},
        3.0: {"pre": 140000, "subsidy": 78000},
        5.0: {"pre": 220000, "subsidy": 78000}
    }

    return {
        "outlook": daily_results,
        "hourly_forecast": hourly_results,
        "comparison": [
            {
                "size": f"{cap}kW", 
                "units": round(totals[cap], 1), 
                "savings": round(totals[cap]*cost, 0),
                "market_price": cost_data[cap]["pre"],
                "pms_subsidy": cost_data[cap]["subsidy"],
                "net_investment": cost_data[cap]["pre"] - cost_data[cap]["subsidy"]
            }
            for cap in capacities
        ]
    }

# --- ROOT UI MOUNT ---
# Serves index.html at root (http://IP/)
app.mount("/", StaticFiles(directory=".", html=True), name="static")