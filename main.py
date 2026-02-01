import os
import joblib
import pandas as pd
import numpy as np
import pvlib
import requests
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv() 
app = FastAPI(title="SolarCast AI - Pincode & City Sync")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL CONSTANTS ---
expected_columns = [
    "Month", "Hour", "Temperature", "Relative Humidity", "Pressure", 
    "Wind Speed", "Solar Zenith Angle", "Clearsky GHI", "lat", "long", 
    "Hour_sin", "Hour_cos", "Month_sin", "Month_cos"
]

# --- ASSET LOADING ---
model = None
pt_y = None

try:
    model_obj = joblib.load('assets/solar_xgboost_model_v2.pkl')
    model = model_obj.get_booster() if hasattr(model_obj, "get_booster") else model_obj
    pt_y = joblib.load('assets/target_transformer_v2.pkl')
    model.feature_names = expected_columns
    print("✅ Assets loaded and synchronized.")
except Exception as e:
    print(f"❌ Load Error: {e}")

# --- DATA MODEL ---
class PincodeRequest(BaseModel):
    pincode: str 

# --- GEOLOCATION HELPER ---
def get_coords_with_city(pincode: str):
    backups = {
        "812001": (25.24, 86.97, "Bhagalpur"),
        "821304": (24.91, 84.18, "Dehri"),
        "800001": (25.59, 85.13, "Patna")
    }
    if pincode in backups: return backups[pincode]
    
    try:
        url = f"https://nominatim.openstreetmap.org/search?postalcode={pincode}&country=India&format=json&addressdetails=1&limit=1"
        res = requests.get(url, headers={'User-Agent': 'SolarCast/2.0'}).json()
        if res:
            address = res[0].get('address', {})
            city_name = address.get('city') or address.get('town') or address.get('village') or address.get('state_district') or "India"
            return float(res[0]['lat']), float(res[0]['lon']), city_name
    except: pass
    return None, None, None

# --- ENDPOINTS ---
@app.post("/predict")
async def predict_solar(req: PincodeRequest):
    try:
        lat, lon, city = get_coords_with_city(req.pincode)
        if lat is None:
            raise HTTPException(status_code=400, detail="Invalid Pincode")

        API_KEY = os.getenv("TOMORROW_API_KEY")
        weather_url = f"https://api.tomorrow.io/v4/weather/realtime?location={lat},{lon}&apikey={API_KEY}"
        response = requests.get(weather_url, timeout=5)
        w_res = response.json()
        
        if 'data' not in w_res:
            vals = {'temperature': 25, 'humidity': 50, 'pressureSurfaceLevel': 1013, 'windSpeed': 5}
            condition = "API Rate Limited (Using Defaults)"
        else:
            vals = w_res['data']['values']
            condition = "Analysis Successful"
        
        ts = pd.Timestamp.now(tz='UTC')
        loc = pvlib.location.Location(lat, lon)
        solpos = loc.get_solarposition(pd.DatetimeIndex([ts]))
        clearsky = loc.get_clearsky(pd.DatetimeIndex([ts]))

        input_data = {
            "Month": int(ts.month), "Hour": int(ts.hour),
            "Temperature": float(vals.get('temperature', 25)),
            "Relative Humidity": float(vals.get('humidity', 50)),
            "Pressure": float(vals.get('pressureSurfaceLevel', 1013)),
            "Wind Speed": float(vals.get('windSpeed', 5)),
            "Solar Zenith Angle": float(solpos['zenith'].iloc[0]),
            "Clearsky GHI": float(clearsky['ghi'].iloc[0]),
            "lat": lat, "long": lon,
            "Hour_sin": np.sin(2 * np.pi * ts.hour / 24),
            "Hour_cos": np.cos(2 * np.pi * ts.hour / 24),
            "Month_sin": np.sin(2 * np.pi * ts.month / 12),
            "Month_cos": np.cos(2 * np.pi * ts.month / 12),
        }

        input_df = pd.DataFrame([input_data])[expected_columns]
        dmatrix = xgb.DMatrix(input_df, feature_names=expected_columns)
        pred_scaled = model.predict(dmatrix)
        pred_raw = pt_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]

        return {
            "prediction": max(0, round(float(pred_raw), 2)),
            "city": city, "lat": lat, "lon": lon,
            "metadata": {"temp": vals.get('temperature'), "condition": condition}
        }
    except Exception as e:
        print(f"❌ Prediction Crash: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/outlook")
async def get_outlook(lat: float, lon: float):
    try:
        API_KEY = os.getenv("TOMORROW_API_KEY")
        url = f"https://api.tomorrow.io/v4/weather/forecast?location={lat},{lon}&apikey={API_KEY}&timesteps=1d"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        days_data = data.get('timelines', {}).get('daily', [])
        location = pvlib.location.Location(lat, lon)
        results = []

        # If API gives no data, trigger fallback immediately
        if not days_data:
            raise ValueError("No daily data from API")

        for day in days_data[:15]:
            date_str = day.get('time').split('T')[0] 
            noon_ts = pd.Timestamp(f"{date_str} 12:00:00", tz='UTC')
            
            clearsky = location.get_clearsky(pd.DatetimeIndex([noon_ts]))
            max_potential = float(clearsky['ghi'].iloc[0])
            
            vals = day.get('values', {})
            cloud_cover = vals.get('cloudCoverAvg', 0) / 100.0
            weather_adjustment = 1.0 - (cloud_cover * 0.6) 
            
            results.append({
                "day": date_str,
                "max_potential": round(max_potential * weather_adjustment, 2)
            })
        return results

    except Exception as e:
        print(f"❌ Outlook Error (Using Fallback): {e}")
        location = pvlib.location.Location(lat, lon)
        start_date = pd.Timestamp.now(tz='UTC').normalize()
        fallback_results = []
        for i in range(15):
            day = start_date + pd.Timedelta(days=i)
            noon_ts = day + pd.Timedelta(hours=12)
            clearsky = location.get_clearsky(pd.DatetimeIndex([noon_ts]))
            fallback_results.append({
                "day": str(day.date()), 
                "max_potential": round(float(clearsky['ghi'].iloc[0]), 2)
            })
        return fallback_results