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
        # Step 1: Attempt to get real 15-day weather data
        url = f"https://api.tomorrow.io/v4/weather/forecast?location={lat},{lon}&apikey={API_KEY}&timesteps=1d"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        # Step 2: Validate the response structure
        if 'timelines' in data and 'daily' in data['timelines']:
            days_data = data['timelines']['daily']
            location = pvlib.location.Location(lat, lon)
            results = []

            for day in days_data[:15]:
                ts = pd.Timestamp(day['time'], tz='UTC')
                # Calculate theoretical physics max
                clearsky = location.get_clearsky(pd.DatetimeIndex([ts]))
                max_potential = float(clearsky['ghi'].iloc[0])
                
                # Step 3: Weather Adjustment (Cloud Cover)
                # Fallback to 0% clouds if the specific field is missing
                cloud_cover = day['values'].get('cloudCoverAvg', 0) / 100.0
                weather_adjustment = 1.0 - (cloud_cover * 0.7) 
                
                results.append({
                    "day": str(ts.date()),
                    "max_potential": round(max(50, max_potential * weather_adjustment), 2)
                })
            return results
        else:
            print("⚠️ Tomorrow.io data format unexpected. Using Fallback.")
            raise ValueError("Incomplete API Data")

    except Exception as e:
        print(f"❌ Outlook Error: {e}")
        # --- SAFETY FALLBACK: Physics-only (Linear) ---
        # This ensures the chart is NEVER 0 even if the API fails
        location = pvlib.location.Location(lat, lon)
        start_date = pd.Timestamp.now(tz='UTC').normalize()
        fallback_results = []
        for i in range(15):
            day = start_date + pd.Timedelta(days=i)
            # Find the highest point of the day for the baseline
            clearsky = location.get_clearsky(pd.date_range(day + pd.Timedelta(hours=6), periods=12, freq='h'))
            fallback_results.append({
                "day": str(day.date()), 
                "max_potential": round(float(clearsky['ghi'].max()), 2)
            })
        return fallback_results