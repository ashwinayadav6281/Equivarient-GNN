import requests
import json
import os
import time

def fetch_fresh_dataset(limit=500):
    dataset_dir = "dataset_equivariant"
    os.makedirs(dataset_dir, exist_ok=True)
    
    output_json = os.path.join(dataset_dir, "elastic_tensors.json")
    
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        print("ERROR: MP_API_KEY environment variable is not set.")
        return
        
    tensors_data = {}
    headers = {"X-API-KEY": api_key}
    
    try:
        print(f"Reading {limit} material IDs from mp_latest_elasticity.csv...")
        valid_mats = []
        
        # Load existing tensors if resuming
        if os.path.exists(output_json):
            with open(output_json, 'r') as f:
                try:
                    tensors_data = json.load(f)
                    print(f"Resuming download. Found {len(tensors_data)} existing tensors.")
                except json.JSONDecodeError:
                    tensors_data = {}
        
        with open("mp_latest_elasticity.csv", "r") as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split(',')
                if parts:
                    mat_id = parts[0]
                    valid_mats.append(mat_id)
                if len(valid_mats) >= limit + 1000:
                    break
        
        print(f"Now fetching structures and tensors for {len(valid_mats)} materials...")
        
        count = len(tensors_data)
        for mat_id in valid_mats:
            if mat_id in tensors_data and os.path.exists(os.path.join(dataset_dir, f"{mat_id}.cif")):
                # Already downloaded, skip
                continue
                
            if count >= limit:
                break
                
            try:
                print(f"Fetching data for {mat_id}...")
                # 1. Fetch elasticity tensor
                el_url = f"https://api.materialsproject.org/materials/elasticity/?material_ids={mat_id}&_fields=elastic_tensor"
                el_resp = requests.get(el_url, headers=headers, timeout=10)
                el_data = el_resp.json().get("data", [])
                
                if not el_data or "elastic_tensor" not in el_data[0]:
                    print(f"  -> No elastic tensor found for {mat_id}")
                    continue
                    
                tensor_obj = el_data[0]["elastic_tensor"]
                if not tensor_obj or not isinstance(tensor_obj, dict) or "ieee_format" not in tensor_obj or not tensor_obj["ieee_format"]:
                    print(f"  -> Invalid tensor format for {mat_id}")
                    continue
                
                # 2. Fetch structure
                struct_url = f"https://api.materialsproject.org/materials/summary/?material_ids={mat_id}&_fields=structure"
                struct_resp = requests.get(struct_url, headers=headers, timeout=10)
                struct_data = struct_resp.json().get("data", [])
                
                if not struct_data or "structure" not in struct_data[0]:
                    print(f"  -> No structure found for {mat_id}")
                    continue
                    
                from pymatgen.core import Structure
                structure = Structure.from_dict(struct_data[0]["structure"])
                
                # Save CIF
                cif_path = os.path.join(dataset_dir, f"{mat_id}.cif")
                structure.to(filename=cif_path, fmt="cif")
                
                # Store tensor
                tensors_data[mat_id] = tensor_obj["ieee_format"]
                count += 1
                print(f"  -> Successfully saved {mat_id} ({count}/{limit})")
                
                if count % 10 == 0:
                    with open(output_json, 'w') as f:
                        json.dump(tensors_data, f)
                
            except Exception as e:
                print(f"  -> Error on {mat_id}: {e}")
                pass
                
        print(f"Successfully saved {count} CIF files to {dataset_dir}/")
        
        # save final tensors to JSON file
        with open(output_json, 'w') as f:
            json.dump(tensors_data, f)
            
        print(f"Saved elastic tensors to {output_json}")
            
    except Exception as e:
        print(f"Error connecting to MP API: {e}")

if __name__ == "__main__":
    fetch_fresh_dataset(500)
