import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 2500  # Updated to 2500 rows

# 1. Identifiers & Categorical Demographics
trip_ids = [f"TRIP-{10000 + i}" for i in range(n_samples)]
emergency_types = np.random.choice(['Cardiac Arrest', 'Severe Trauma', 'Respiratory Failure', 'Stroke', 'General Emergency'], size=n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1])
hospitals = ['MY Hospital', 'Bombay Hospital Indore', 'CHL Hospital', 'Apollo Rajshree', 'Medanta Indore', 'Choithram Hospital']
assigned_hospitals = np.random.choice(hospitals, size=n_samples)

# 2. Vital Signs & Clinical Triage
heart_rate = np.random.normal(loc=95, scale=22, size=n_samples).clip(40, 180).round(1)
spo2 = np.random.normal(loc=93, scale=4.5, size=n_samples).clip(70, 100).round(1)
news2_score = np.random.randint(0, 12, size=n_samples)

severity_levels = []
for score in news2_score:
    if score <= 2: severity_levels.append('Low')
    elif score <= 4: severity_levels.append('Medium')
    elif score <= 6: severity_levels.append('High')
    else: severity_levels.append('Critical')

# 3. Traffic, Road & Routing Factors
distance_km = np.random.uniform(2.5, 22.0, size=n_samples).round(2)
speed_limit_kmh = np.random.choice([40, 50, 60, 80], size=n_samples)
traffic_density_index = np.random.uniform(0.1, 1.0, size=n_samples).round(2)
accident_risk_index = np.random.uniform(0.05, 0.95, size=n_samples).round(2)

base_time_min = (distance_km / speed_limit_kmh) * 60
actual_travel_time_min = (base_time_min + (base_time_min * traffic_density_index * 1.2) + np.random.normal(0, 1.2, n_samples)).clip(3, 60).round(1)

# 4. Hospital Capacity & System Risk
icu_beds_available = np.random.randint(0, 15, size=n_samples)
er_queue_count = np.random.randint(1, 20, size=n_samples)
risk_score = (news2_score * 0.4 + traffic_density_index * 3 + (15 - icu_beds_available) * 0.3).clip(1, 10).round(2)
reroute_triggered = np.where((severity_levels == 'Critical') | (risk_score > 7.5), 'Yes', 'No')

# Assemble Clean Dataframe
df = pd.DataFrame({
    'Trip_ID': trip_ids,
    'Emergency_Type': emergency_types,
    'Assigned_Hospital': assigned_hospitals,
    'Severity': severity_levels,
    'NEWS2_Score': news2_score,
    'Heart_Rate_BPM': heart_rate,
    'SpO2_Percent': spo2,
    'Distance_KM': distance_km,
    'Speed_Limit_KMH': speed_limit_kmh,
    'Traffic_Density_Index': traffic_density_index,
    'Accident_Risk_Index': accident_risk_index,
    'Actual_Travel_Time_Min': actual_travel_time_min,
    'ICU_Beds_Available': icu_beds_available,
    'ER_Queue_Count': er_queue_count,
    'Risk_Score': risk_score,
    'Reroute_Triggered': reroute_triggered
})

df.to_csv('emergency_trip_telematics.csv', index=False)
print("Successfully generated emergency_trip_telematics.csv (2,500 rows, 16 essential attributes)")