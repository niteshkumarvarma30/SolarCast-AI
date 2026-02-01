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
app = FastAPI(title="SolarCast AI - 2026 Production Edition")

# --- CORS CONFIGURATION ---
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

# --- UPDATED DATA MODEL ---
class SolarFinancialRequest(BaseModel):
    pincode: str
    avg_daily_usage: float 
    unit_cost: float        

# --- GEOLOCATION HELPER ---
def get_coords_with_city(pincode: str):
    backups = {"812001": (25.24, 86.97, "Bhagalpur"), "813210": (25.303, 86.967, "Bhagalpur"), "821304": (24.91, 84.18, "Dehri"), "800001": (25.59, 85.13, "Patna")}
    if pincode in backups: return backups[pincode]
    try:
        url = f"https://nominatim.openstreetmap.org/search?postalcode={pincode}&country=India&format=json&addressdetails=1&limit=1"
        res = requests.get(url, headers={'User-Agent': 'SolarCast/2.0'}).json()
        if res:
            address = res[0].get('address', {})
            city = address.get('city') or address.get('town') or address.get('village') or "India"
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
        res = requests.get(weather_url, timeout=5)
        w_res = res.json()
        
        # API LIMIT DETECTION
        if 'data' not in w_res:
            vals = {'temperature': 25, 'humidity': 50, 'pressureSurfaceLevel': 1013, 'windSpeed': 5}
            condition = "API LIMIT REACHED (Using Physics Fallback)"
        else:
            vals = w_res['data']['values']
            condition = "Analysis Successful"
        
        ts = pd.Timestamp.now(tz='UTC')
        loc = pvlib.location.Location(lat, lon)
        solpos = loc.get_solarposition(pd.DatetimeIndex([ts]))
        clearsky = loc.get_clearsky(pd.DatetimeIndex([ts]))

        input_data = {"Month": ts.month, "Hour": ts.hour, "Temperature": vals.get('temperature'), "Relative Humidity": vals.get('humidity'), "Pressure": vals.get('pressureSurfaceLevel'), "Wind Speed": vals.get('windSpeed'), "Solar Zenith Angle": solpos['zenith'].iloc[0], "Clearsky GHI": clearsky['ghi'].iloc[0], "lat": lat, "long": lon, "Hour_sin": np.sin(2*np.pi*ts.hour/24), "Hour_cos": np.cos(2*np.pi*ts.hour/24), "Month_sin": np.sin(2*np.pi*ts.month/12), "Month_cos": np.cos(2*np.pi*ts.month/12)}
        
        dmatrix = xgb.DMatrix(pd.DataFrame([input_data])[expected_columns], feature_names=expected_columns)
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
async def get_outlook(lat: float, lon: float):
    # FORCED NOON-ONLY LOGIC for 15-day stable outlook
    location = pvlib.location.Location(lat, lon)
    start_date = pd.Timestamp.now(tz='UTC').normalize()
    results = []
    for i in range(15):
        day = start_date + pd.Timedelta(days=i)
        noon_ts = day + pd.Timedelta(hours=12) # TARGET 12:00 PM
        clearsky = location.get_clearsky(pd.DatetimeIndex([noon_ts]))
        results.append({"day": str(day.date()), "max_potential": round(float(clearsky['ghi'].iloc[0]), 2)})
    return results