"""
Full dataset download script for the server.
Downloads elastic tensors + CIF structures from Materials Project
for ALL ~13,000 materials.

Based on fetch_elastic_tensors.py — scaled up to the full dataset.

Run on server:
    export MP_API_KEY="your_api_key_here"
    python download_dataset.py

Has resume support — re-run safely if interrupted.
Takes ~2-4 hours for the full 13k dataset.
"""
import os
import sys
import json
import time
import requests
import pandas as pd
from pymatgen.core import Structure

# ── Configuration ──
API_KEY = os.environ.get("MP_API_KEY", "8UjTxRqKoojsABsTyt6CQULQce6U5Ctk")
HEADERS = {"accept": "application/json", "X-API-KEY": API_KEY}

# Directories
CGCNN_DIR = "cgcnn_dataset_full"          # CIF files + id_prop CSVs for CGCNN
EQUIVARIANT_DIR = "dataset_equivariant"    # CIF files + elastic_tensors.json
CSV_FILE = "mp_latest_elasticity.csv"


def step1_fetch_elasticity_data():
    """Download bulk & shear modulus values from Materials Project."""
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        print(f"[Step 1] SKIP — {CSV_FILE} already exists ({len(df)} materials)")
        return df

    print("[Step 1] Fetching elasticity data from Materials Project API...")
    url = "https://api.materialsproject.org/materials/elasticity/"
    data = []
    skip = 0
    limit = 1000

    while True:
        print(f"  Downloading batch (skip={skip})...")
        params = {
            "_fields": "material_id,formula_pretty,bulk_modulus,shear_modulus",
            "_limit": limit,
            "_skip": skip,
        }
        res = requests.get(url, headers=HEADERS, params=params)

        if res.status_code != 200:
            print(f"  API Error {res.status_code}: {res.text}")
            break

        json_data = res.json().get("data", [])
        if not json_data:
            break

        for item in json_data:
            bm = item.get("bulk_modulus") or {}
            sm = item.get("shear_modulus") or {}
            k = bm.get("vrh")
            g = sm.get("vrh")
            if k is not None and g is not None:
                data.append({
                    "material_id": item.get("material_id"),
                    "formula": item.get("formula_pretty", "Unknown"),
                    "bulk_modulus_vrh": k,
                    "shear_modulus_vrh": g,
                })

        skip += limit
        time.sleep(0.5)

    df = pd.DataFrame(data)
    df.to_csv(CSV_FILE, index=False)
    print(f"  Saved {len(df)} materials to {CSV_FILE}")
    return df


def step2_create_cgcnn_target_csvs(df):
    """Create id_prop_bulk.csv and id_prop_shear.csv for CGCNN."""
    os.makedirs(CGCNN_DIR, exist_ok=True)

    bulk_path = os.path.join(CGCNN_DIR, "id_prop_bulk.csv")
    shear_path = os.path.join(CGCNN_DIR, "id_prop_shear.csv")

    df[["material_id", "bulk_modulus_vrh"]].to_csv(bulk_path, index=False, header=False)
    df[["material_id", "shear_modulus_vrh"]].to_csv(shear_path, index=False, header=False)

    print(f"[Step 2] Created CGCNN target CSVs ({len(df)} entries each)")


def step3_ensure_atom_init():
    """Make sure atom_init.json exists in the CGCNN dataset directory."""
    atom_init_dst = os.path.join(CGCNN_DIR, "atom_init.json")
    if os.path.exists(atom_init_dst):
        print(f"[Step 3] atom_init.json already exists")
        return

    # Try to find it elsewhere in the repo
    candidates = [
        os.path.join("cgcnn", "data", "sample-regression", "atom_init.json"),
        os.path.join(EQUIVARIANT_DIR, "atom_init.json"),
    ]
    for src in candidates:
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, atom_init_dst)
            print(f"[Step 3] Copied atom_init.json from {src}")
            return

    print(f"[Step 3] ERROR: atom_init.json not found anywhere!")
    sys.exit(1)


def step4_download_structures_and_tensors(df):
    """
    Download CIF structures and full elastic tensors for all materials.
    
    Saves CIF files to BOTH directories (cgcnn_dataset_full/ and dataset_equivariant/)
    and elastic tensors to dataset_equivariant/elastic_tensors.json.
    
    Based on fetch_elastic_tensors.py but scaled to full dataset with batching.
    """
    os.makedirs(CGCNN_DIR, exist_ok=True)
    os.makedirs(EQUIVARIANT_DIR, exist_ok=True)

    mp_ids = df["material_id"].tolist()

    # ── Load existing progress (resume support) ──
    tensor_json_path = os.path.join(EQUIVARIANT_DIR, "elastic_tensors.json")
    if os.path.exists(tensor_json_path):
        with open(tensor_json_path, "r") as f:
            try:
                tensors_data = json.load(f)
            except json.JSONDecodeError:
                tensors_data = {}
    else:
        tensors_data = {}

    # Check which CIFs already exist in both dirs
    existing_cgcnn = set(
        f.replace(".cif", "") for f in os.listdir(CGCNN_DIR) if f.endswith(".cif")
    )
    existing_equi = set(
        f.replace(".cif", "") for f in os.listdir(EQUIVARIANT_DIR) if f.endswith(".cif")
    )

    # A material is "done" only if it has CIF in both dirs AND tensor data
    done = existing_cgcnn & existing_equi & set(tensors_data.keys())
    remaining = [mid for mid in mp_ids if mid not in done]

    print(f"[Step 4] Downloading structures + elastic tensors...")
    print(f"  Total materials:      {len(mp_ids)}")
    print(f"  Already complete:     {len(done)}")
    print(f"  Remaining to fetch:   {len(remaining)}")

    if not remaining:
        print("  All data already downloaded!")
        return tensors_data

    success = 0
    fail = 0
    chunk_size = 50

    for i in range(0, len(remaining), chunk_size):
        chunk = remaining[i : i + chunk_size]
        progress = min(i + chunk_size, len(remaining))
        print(f"  [{progress}/{len(remaining)}] Fetching batch...", end="", flush=True)

        # ── Fetch elastic tensors for this batch ──
        try:
            el_url = "https://api.materialsproject.org/materials/elasticity/"
            el_params = {
                "material_ids": ",".join(chunk),
                "_fields": "material_id,elastic_tensor",
            }
            el_resp = requests.get(el_url, headers=HEADERS, params=el_params, timeout=30)
            
            if el_resp.status_code != 200:
                print(f" tensor API error ({el_resp.status_code})")
                fail += len(chunk)
                time.sleep(1)
                continue

            el_data = {
                item["material_id"]: item.get("elastic_tensor")
                for item in el_resp.json().get("data", [])
                if item.get("elastic_tensor")
            }
        except Exception as e:
            print(f" tensor error: {e}")
            fail += len(chunk)
            time.sleep(1)
            continue

        # ── Fetch structures for this batch ──
        try:
            struct_url = "https://api.materialsproject.org/materials/summary/"
            struct_params = {
                "material_ids": ",".join(chunk),
                "_fields": "material_id,structure",
            }
            struct_resp = requests.get(struct_url, headers=HEADERS, params=struct_params, timeout=30)

            if struct_resp.status_code != 200:
                print(f" structure API error ({struct_resp.status_code})")
                fail += len(chunk)
                time.sleep(1)
                continue

            struct_data = {
                item["material_id"]: item.get("structure")
                for item in struct_resp.json().get("data", [])
                if item.get("structure")
            }
        except Exception as e:
            print(f" structure error: {e}")
            fail += len(chunk)
            time.sleep(1)
            continue

        # ── Save CIF files + tensor data ──
        batch_ok = 0
        for mat_id in chunk:
            struct_dict = struct_data.get(mat_id)
            tensor_obj = el_data.get(mat_id)

            if not struct_dict:
                continue

            try:
                structure = Structure.from_dict(struct_dict)

                # Save CIF to cgcnn_dataset_full/
                cif_cgcnn = os.path.join(CGCNN_DIR, f"{mat_id}.cif")
                if not os.path.exists(cif_cgcnn):
                    structure.to(filename=cif_cgcnn)

                # Save CIF to dataset_equivariant/
                cif_equi = os.path.join(EQUIVARIANT_DIR, f"{mat_id}.cif")
                if not os.path.exists(cif_equi):
                    structure.to(filename=cif_equi)

                # Store elastic tensor (IEEE format)
                if tensor_obj and isinstance(tensor_obj, dict):
                    ieee = tensor_obj.get("ieee_format")
                    if ieee:
                        tensors_data[mat_id] = ieee

                batch_ok += 1
            except Exception as e:
                pass  # Skip problematic structures

        success += batch_ok
        print(f" OK ({batch_ok} saved)")

        # Save tensor JSON every 200 materials (checkpoint)
        if (i // chunk_size) % 4 == 0 and tensors_data:
            with open(tensor_json_path, "w") as f:
                json.dump(tensors_data, f)

        time.sleep(0.5)

    # Final save of tensor data
    with open(tensor_json_path, "w") as f:
        json.dump(tensors_data, f)

    print(f"\n  Download complete!")
    print(f"  New downloads: {success}")
    print(f"  Elastic tensors saved: {len(tensors_data)}")
    if fail:
        print(f"  Failed batches: {fail} (re-run to retry)")

    return tensors_data


def step5_verify():
    """Final verification that everything is ready."""
    print(f"\n{'='*60}")
    print(f"  DATASET VERIFICATION")
    print(f"{'='*60}")

    cgcnn_cifs = len([f for f in os.listdir(CGCNN_DIR) if f.endswith(".cif")])
    equi_cifs = len([f for f in os.listdir(EQUIVARIANT_DIR) if f.endswith(".cif")])

    tensor_path = os.path.join(EQUIVARIANT_DIR, "elastic_tensors.json")
    tensor_count = 0
    if os.path.exists(tensor_path):
        with open(tensor_path) as f:
            tensor_count = len(json.load(f))

    checks = [
        ("cgcnn_dataset_full/ CIF files", f"{cgcnn_cifs}", cgcnn_cifs > 0),
        ("cgcnn_dataset_full/ id_prop_bulk.csv",
         "OK" if os.path.exists(os.path.join(CGCNN_DIR, "id_prop_bulk.csv")) else "MISSING",
         os.path.exists(os.path.join(CGCNN_DIR, "id_prop_bulk.csv"))),
        ("cgcnn_dataset_full/ id_prop_shear.csv",
         "OK" if os.path.exists(os.path.join(CGCNN_DIR, "id_prop_shear.csv")) else "MISSING",
         os.path.exists(os.path.join(CGCNN_DIR, "id_prop_shear.csv"))),
        ("cgcnn_dataset_full/ atom_init.json",
         "OK" if os.path.exists(os.path.join(CGCNN_DIR, "atom_init.json")) else "MISSING",
         os.path.exists(os.path.join(CGCNN_DIR, "atom_init.json"))),
        ("dataset_equivariant/ CIF files", f"{equi_cifs}", equi_cifs > 0),
        ("dataset_equivariant/ elastic_tensors.json", f"{tensor_count} tensors", tensor_count > 0),
    ]

    all_ok = True
    for name, status, ok in checks:
        icon = "✓" if ok else "✗"
        print(f"  {icon} {name}: {status}")
        if not ok:
            all_ok = False

    if all_ok:
        print(f"\n  Dataset is ready! Run:  python train_server.py")
    else:
        print(f"\n  Some files missing. Re-run this script to retry.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("  CGCNN + Equivariant Full Dataset Download")
    print("=" * 60)
    print()

    df = step1_fetch_elasticity_data()
    step2_create_cgcnn_target_csvs(df)
    step3_ensure_atom_init()
    step4_download_structures_and_tensors(df)
    step5_verify()
