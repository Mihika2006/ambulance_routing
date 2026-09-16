import osmnx as ox
import pandas as pd

# 1. Fetch real hospitals in Indore from OpenStreetMap
hospitals_gdf = ox.features.features_from_place("Indore, Madhya Pradesh, India", tags={"amenity": "hospital"})

# 2. Extract Latitude, Longitude, and Name
hospitals_df = pd.DataFrame({
    'Hospital_ID': [f"HOSP-{100 + i}" for i in range(len(hospitals_gdf))],
    'Name': hospitals_gdf['name'].fillna('Unnamed Hospital Facility'),
    'Latitude': hospitals_gdf.geometry.centroid.y,
    'Longitude': hospitals_gdf.geometry.centroid.x,
    'Emergency_Type': ['General / Trauma'] * len(hospitals_gdf), # Default base capability
    'Bed_Capacity': [50] * len(hospitals_gdf) # Default capacity baseline
})

# 3. Save to CSV
hospitals_df.to_csv("indore_hospitals.csv", index=False)
print(f"Saved {len(hospitals_df)} real Indore hospitals to indore_hospitals.csv")