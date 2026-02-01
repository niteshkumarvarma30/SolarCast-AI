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
# Moved here so it's globally accessible to all endpoints
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
    # Use global keywords to update the variables defined above
    model = model_obj.get_booster() if hasattr(model_obj, "get_booster") else model_obj
    pt_y = joblib.load('assets/target_transformer_v2.pkl')
    
    # Apply feature names to the booster
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
    if model is None or pt_y is None:
        raise HTTPException(status_code=500, detail="Model assets not loaded")
        
    try:
        lat, lon, city = get_coords_with_city(req.pincode)
        if lat is None:
            raise HTTPException(status_code=400, detail="Invalid Pincode")

        API_KEY = os.getenv("TOMORROW_API_KEY")
        weather_url = f"https://api.tomorrow.io/v4/weather/realtime?location={lat},{lon}&apikey={API_KEY}"
        w_res = requests.get(weather_url).json()
        vals = w_res['data']['values']
        
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
            "city": city,
            "lat": lat, "lon": lon,
            "metadata": {"temp": vals.get('temperature'), "condition": "Analysis Successful"}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/outlook")
async def get_outlook(lat: float, lon: float):
    try:
        API_KEY = os.getenv("TOMORROW_API_KEY")
        url = f"https://api.tomorrow.io/v4/weather/forecast?location={lat},{lon}&apikey={API_KEY}&timesteps=1d"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        # Check for the correct nested path in the new API version
        timelines = data.get('timelines', {})
        daily_data = timelines.get('daily', [])

        if not daily_data:
            print("⚠️ No daily data found in API response. Using physics fallback.")
            raise ValueError("Empty API response")

        location = pvlib.location.Location(lat, lon)
        results = []

        for day in daily_data[:15]:
            # Convert ISO string to timestamp
            ts = pd.Timestamp(day.get('time'), tz='UTC')
            
            # Physics Baseline (Clearsky)
            clearsky = location.get_clearsky(pd.DatetimeIndex([ts]))
            max_potential = float(clearsky['ghi'].iloc[0])
            
            # Extract weather values safely
            vals = day.get('values', {})
            # Try multiple common cloud keys
            cloud = vals.get('cloudCoverAvg') or vals.get('cloudCover') or 0
            
            # Realistic Damping: Even on cloudy days, GHI is rarely 0. 
            # We use a 0.3 - 1.0 multiplier range.
            weather_multiplier = 1.0 - (min(cloud, 100) / 100.0 * 0.7)
            
            results.append({
                "day": str(ts.date()),
                "max_potential": round(max_potential * weather_multiplier, 2)
            })
        return results

    except Exception as e:
        print(f"❌ Outlook Logic Error: {e}")
        # Final Fallback to Clearsky Physics so the chart is NEVER flat
        location = pvlib.location.Location(lat, lon)
        start_date = pd.Timestamp.now(tz='UTC').normalize()
        return [{"day": str((start_date + pd.Timedelta(days=i)).date()), 
                 "max_potential": round(float(location.get_clearsky(pd.date_range(start_date + pd.Timedelta(days=i), periods=1, freq='h'))['ghi'].max()), 2)} 
                for i in range(15)]