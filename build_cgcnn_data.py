import os
import time
import requests
import pandas as pd
from pymatgen.core import Structure

API_KEY = "8UjTxRqKoojsABsTyt6CQULQce6U5Ctk"
INPUT_CSV = "mp_latest_elasticity.csv" # Change if you named it differently
OUTPUT_DIR = "cgcnn_dataset_full"

def build_dataset():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Could not find '{INPUT_CSV}'.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    
    print("--- STEP 1: Generating Pipeline Target Files ---")
    # 1. Generate the Bulk Modulus target file (No headers, 2 columns)
    bulk_df = df[["material_id", "bulk_modulus_vrh"]]
    bulk_df.to_csv(os.path.join(OUTPUT_DIR, "id_prop_bulk.csv"), index=False, header=False)
    
    # 2. Generate the Shear Modulus target file (No headers, 2 columns)
    shear_df = df[["material_id", "shear_modulus_vrh"]]
    shear_df.to_csv(os.path.join(OUTPUT_DIR, "id_prop_shear.csv"), index=False, header=False)
    
    print(f"Created id_prop_bulk.csv and id_prop_shear.csv in '{OUTPUT_DIR}/'")

    print("\n--- STEP 2: Downloading 3D .cif Structures ---")
    headers = {"accept": "application/json", "X-API-KEY": API_KEY}
    url = "https://api.materialsproject.org/materials/summary/"
    
    # Extract all the IDs from your CSV
    mp_ids = df["material_id"].tolist()
    chunk_size = 50 # Download 50 at a time to prevent URL length limits
    
    success_count = 0
    
    for i in range(0, len(mp_ids), chunk_size):
        chunk = mp_ids[i:i+chunk_size]
        print(f"Fetching structures {i+1} to {min(i+chunk_size, len(mp_ids))} of {len(mp_ids)}...")
        
        params = {
            "material_ids": ",".join(chunk),
            "_fields": "material_id,structure"
        }
        
        try:
            res = requests.get(url, headers=headers, params=params)
            if res.status_code == 200:
                json_data = res.json().get("data", [])
                
                for item in json_data:
                    mat_id = item.get("material_id")
                    struct_dict = item.get("structure")
                    
                    if struct_dict:
                        # Convert the raw JSON dictionary into a 3D pymatgen structure
                        struct = Structure.from_dict(struct_dict)
                        # Save it as a .cif file in the dataset folder
                        struct.to(filename=os.path.join(OUTPUT_DIR, f"{mat_id}.cif"))
                        success_count += 1
            else:
                print(f"Chunk failed with status {res.status_code}: {res.text}")
                
        except Exception as e:
            print(f"Error processing chunk: {e}")
            
        # Small delay to keep the server happy
        time.sleep(0.5)

    print("\n=== DATASET COMPLETE ===")
    print(f"Successfully downloaded {success_count} .cif files into '{OUTPUT_DIR}/'!")

if __name__ == "__main__":
    build_dataset()