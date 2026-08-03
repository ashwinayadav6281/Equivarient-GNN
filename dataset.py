import requests
import pandas as pd
import time

API_KEY = "8UjTxRqKoojsABsTyt6CQULQce6U5Ctk"
headers = {"accept": "application/json", "X-API-KEY": API_KEY}

# Hitting the raw REST API directly, bypassing the broken python client
url = "https://api.materialsproject.org/materials/elasticity/"

print("Fetching raw data directly from Materials Project servers...")

data = []
skip = 0
limit = 1000 # Download in batches of 1000 to be safe and fast

while True:
    print(f"Downloading batch (skip={skip})...")
    params = {
        "_fields": "material_id,formula_pretty,bulk_modulus,shear_modulus",
        "_limit": limit,
        "_skip": skip
    }
    
    # Send the raw web request
    res = requests.get(url, headers=headers, params=params)
    
    if res.status_code != 200:
        print(f"API Error {res.status_code}: {res.text}")
        break
        
    # Extract the raw JSON dictionary
    json_data = res.json().get("data", [])
    
    # If the batch is empty, we've reached the end of the database
    if not json_data:
        break
        
    for item in json_data:
        mat_id = item.get("material_id")
        formula = item.get("formula_pretty", "Unknown")
        
        # Safely extract the dictionaries
        bm = item.get("bulk_modulus") or {}
        sm = item.get("shear_modulus") or {}
        
        # Grab the VRH values directly from the JSON text
        k = bm.get("vrh")
        g = sm.get("vrh")
        
        if k is not None and g is not None:
            data.append({
                "material_id": mat_id,
                "formula": formula,
                "bulk_modulus_vrh": k,
                "shear_modulus_vrh": g
            })
            
    skip += limit
    # A tiny pause so we don't get blocked by the server
    time.sleep(0.5) 

# Save directly to CSV
df = pd.DataFrame(data)
csv_filename = "mp_latest_elasticity.csv"
df.to_csv(csv_filename, index=False)

print(f"\n=== SUCCESS! ===")
print(f"Saved {len(df)} valid materials to {csv_filename}")