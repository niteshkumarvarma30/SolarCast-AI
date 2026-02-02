☀️ SolarCast AI: Enterprise Energy Decision System (2026 Edition)
SolarCast AI is an end-to-end Machine Learning solution designed for the PM Surya Ghar: Muft Bijli Yojana (2026). It solves the "Data Opacity" problem by translating raw meteorological forecasts into actionable financial insights (Units generated & Revenue saved) using a Dual-Engine Architecture.


LIVE Link : http://16.16.131.10



🏗️ 1. Exploratory Data Analysis (EDA) & Challenges
A standard ML model fails on solar data because of its high non-linearity and sensitivity to seasonal variance.

Challenges Faced:
Data Skewness: Irradiance data is heavily right-skewed (mostly zeros at night).

Non-Linear Patterns: Solar energy follows a diurnal (daily) and seasonal (yearly) cycle that standard linear models cannot capture.

Outliers: Sensor noise during thunderstorms often created false spikes in training data.

The Solution (The ML Engineer Approach):
Cyclical Feature Engineering: I encoded Hour and Month using Trigonometric Sine/Cosine Transforms. This ensures the model treats 23:00 and 01:00 as "close" rather than distant integers.

Clearsky Baseline: I integrated pvlib to calculate the theoretical Global Horizontal Irradiance (GHI). By adding this as a feature, the model focuses on the residuals (the difference caused by actual weather) rather than learning simple day/night patterns from scratch.



🤖 2. Model Training & Optimization
The "brain" of the system is a highly tuned XGBoost v2.0 Regressor.

The Stack:
Algorithm: XGBoost (Extreme Gradient Boosting).

Target Transformation: Used a PowerTransformer (Yeo-Johnson) to normalize the energy output, significantly improving the Mean Absolute Error (MAE).

Cross-Validation: Time-Series Split (ensures we don't use "future" data to predict "past" data).

Why XGBoost?
XGBoost handles missing values (common in weather sensors) natively and outperforms Deep Learning models on structured tabular data for real-time inference on low-resource EC2 instances.



🛰️ 3. Deployment & MLOps Strategy
Moving from a Jupyter Notebook to a Global Enterprise Portal required a robust deployment pipeline.

The Infrastructure:
Hosting: AWS EC2 (Ubuntu 22.04 LTS).

Server Stack: FastAPI + Uvicorn + PM2 (Process Management).

Static Serving: Configured FastAPI to mount the root directory, serving the UI directly at the public IP for zero-latency access from any device.

Global Portability Challenge:
Obstacle: Localhost URLs break on mobile devices.

Solution: Implemented Public IP Mapping. By hardcoding the AWS Static IP into the frontend fetch logic and opening Port 80, the dashboard is accessible via 4G/5G mobile data from any location worldwide.



🛡️ 4. Handling Data Drift: The Physics Fallback
In production, weather APIs often fail or data patterns shift due to unusual smog/dust storms (Data Drift).

The Problem:
If the AI encounters a weather pattern it hasn't seen (e.g., intense 2026 winter haze), its accuracy drops, or it might error out if the API is offline.
The Tactic: Physics-Based Fallback Engine
I implemented a Deterministic Geometry Engine that activates if AI confidence is low:
  The Math:
      It calculates the Solar Zenith Angle ($\theta_z$) based on the Pincode's Latitude/Longitude:
       
       cos(theta_z) = sin(phi)sin(delta) + cos(phi)cos(delta)cos(omega)

The Benefit: Since the sun's position is a mathematical constant, the dashboard never shows an error. It provides a "Clear Sky" baseline, ensuring 100% system reliability for the end-user. 



🌍 5. Real-Life Solution & Social Impact
SolarCast AI directly aligns with the PM Surya Ghar 2026 goals:

System Scaling: Users can compare 1kW, 3kW, and 5kW systems instantly.

Subsidy Logic: It calculates the Net Investment after government grants (₹30,000 to ₹78,000).

Outcome: Translates 16.8 kWh of "winter generation" into ₹100+ savings, proving the financial case for solar adoption in India.



💻 6. Installation & Setup

1)Clone the Repo:

Bash
git clone https://github.com/yourusername/SolarCast-AI.git

2)Environment Setup:

Bash
pip install -r requirements.txt

3)Run Locally:

Bash
uvicorn main:app --host 0.0.0.0 --port 80



A Message to Recruiters :

"I built SolarCast AI not just to predict numbers, but to solve the technical and economic hurdles of renewable energy adoption. From handling data skew with Sin/Cos transforms to ensuring 100% uptime with Physics Fallbacks on AWS, this project showcases my ability to deliver production-ready AI."

