from mp_api.client import MPRester
import pandas as pd

API_KEY = "8UjTxRqKoojsABsTyt6CQULQce6U5Ctk"

print("Forcing live database query to bypass S3 servers...")
with MPRester(API_KEY) as mpr:
    # By specifying a range for k_vrh, we bypass the bulk-download bug 
    # and force the API to actively retrieve the data fields.
    docs = mpr.materials.summary.search(
        k_vrh=(0, 20000), 
        fields=["material_id", "formula_pretty", "bulk_modulus", "shear_modulus"]
    )

print("Parsing data...")
data = []
for doc in docs:
    try:
        # Safely fetch the properties
        bm = getattr(doc, "bulk_modulus", None)
        sm = getattr(doc, "shear_modulus", None)
        
        if bm and sm:
            # Handle whether the API returns a dictionary or an object
            k = bm.get("vrh") if isinstance(bm, dict) else getattr(bm, "vrh", None)
            g = sm.get("vrh") if isinstance(sm, dict) else getattr(sm, "vrh", None)
            
            if k is not None and g is not None:
                data.append({
                    "material_id": str(getattr(doc, "material_id", "")),
                    "formula": str(getattr(doc, "formula_pretty", "")),
                    "bulk_modulus_vrh": k,
                    "shear_modulus_vrh": g
                })
    except Exception as e:
        continue

df = pd.DataFrame(data)
csv_filename = "mp_latest_elasticity.csv"
df.to_csv(csv_filename, index=False)

print(f"\n=== SUCCESS! ===")
print(f"Saved {len(df)} valid materials to {csv_filename}")