"""
Fetch elastic tensors + CIF structures for ALL materials in mp_latest_elasticity.csv.

This is the server-scaled version of fetch_elastic_tensors.py:
  - Original: 500 materials, one-by-one API calls
  - This:     ALL ~13,000 materials, batched API calls (50x faster)

Usage:
    export MP_API_KEY="your_key_here"
    python fetch_elastic_tensors_full.py

Outputs to dataset_equivariant/:
    - elastic_tensors.json  (material_id → 6x6 IEEE tensor)
    - *.cif files           (crystal structures)

Has resume support — safe to re-run if interrupted.
"""
import os
import sys
import json
import time
import requests
from pymatgen.core import Structure

API_KEY = os.environ.get("MP_API_KEY", "8UjTxRqKoojsABsTyt6CQULQce6U5Ctk")
HEADERS = {"accept": "application/json", "X-API-KEY": API_KEY}
DATASET_DIR = "dataset_equivariant"
CSV_FILE = "mp_latest_elasticity.csv"
TENSOR_JSON = os.path.join(DATASET_DIR, "elastic_tensors.json")


def load_material_ids():
    """Read all material IDs from the elasticity CSV."""
    if not os.path.exists(CSV_FILE):
        print(f"ERROR: {CSV_FILE} not found. Run dataset.py first.")
        sys.exit(1)

    ids = []
    with open(CSV_FILE, "r") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split(",")
            if parts:
                ids.append(parts[0])
    return ids


def load_existing_progress():
    """Load already-downloaded tensor data (resume support)."""
    if os.path.exists(TENSOR_JSON):
        with open(TENSOR_JSON, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def save_tensors(tensors_data):
    """Save tensor data to JSON (checkpoint)."""
    with open(TENSOR_JSON, "w") as f:
        json.dump(tensors_data, f)


def fetch_all(batch_size=50):
    os.makedirs(DATASET_DIR, exist_ok=True)

    mp_ids = load_material_ids()
    tensors_data = load_existing_progress()

    # Find which materials still need downloading
    existing_cifs = set(
        f.replace(".cif", "") for f in os.listdir(DATASET_DIR) if f.endswith(".cif")
    )
    done = existing_cifs & set(tensors_data.keys())
    remaining = [mid for mid in mp_ids if mid not in done]

    print(f"Total materials in CSV:   {len(mp_ids)}")
    print(f"Already downloaded:       {len(done)}")
    print(f"Remaining to fetch:       {len(remaining)}")
    print()

    if not remaining:
        print("All data already downloaded!")
        return

    success_total = 0
    fail_total = 0

    for i in range(0, len(remaining), batch_size):
        chunk = remaining[i : i + batch_size]
        progress = min(i + batch_size, len(remaining))
        print(f"[{progress}/{len(remaining)}] Fetching {len(chunk)} materials...", end="", flush=True)

        # ── 1. Fetch elastic tensors (batched) ──
        try:
            el_resp = requests.get(
                "https://api.materialsproject.org/materials/elasticity/",
                headers=HEADERS,
                params={
                    "material_ids": ",".join(chunk),
                    "_fields": "material_id,elastic_tensor",
                },
                timeout=30,
            )
            if el_resp.status_code != 200:
                print(f" tensor API error ({el_resp.status_code})")
                fail_total += len(chunk)
                time.sleep(2)
                continue

            el_map = {}
            for item in el_resp.json().get("data", []):
                tensor_obj = item.get("elastic_tensor")
                if tensor_obj and isinstance(tensor_obj, dict) and tensor_obj.get("ieee_format"):
                    el_map[item["material_id"]] = tensor_obj["ieee_format"]

        except Exception as e:
            print(f" tensor error: {e}")
            fail_total += len(chunk)
            time.sleep(2)
            continue

        # ── 2. Fetch structures (batched) ──
        try:
            struct_resp = requests.get(
                "https://api.materialsproject.org/materials/summary/",
                headers=HEADERS,
                params={
                    "material_ids": ",".join(chunk),
                    "_fields": "material_id,structure",
                },
                timeout=30,
            )
            if struct_resp.status_code != 200:
                print(f" structure API error ({struct_resp.status_code})")
                fail_total += len(chunk)
                time.sleep(2)
                continue

            struct_map = {}
            for item in struct_resp.json().get("data", []):
                if item.get("structure"):
                    struct_map[item["material_id"]] = item["structure"]

        except Exception as e:
            print(f" structure error: {e}")
            fail_total += len(chunk)
            time.sleep(2)
            continue

        # ── 3. Save CIF files + store tensor data ──
        batch_ok = 0
        for mat_id in chunk:
            if mat_id not in el_map or mat_id not in struct_map:
                continue
            try:
                structure = Structure.from_dict(struct_map[mat_id])
                cif_path = os.path.join(DATASET_DIR, f"{mat_id}.cif")
                if not os.path.exists(cif_path):
                    structure.to(filename=cif_path, fmt="cif")
                tensors_data[mat_id] = el_map[mat_id]
                batch_ok += 1
            except Exception:
                pass

        success_total += batch_ok
        print(f" OK ({batch_ok} saved, {len(tensors_data)} total tensors)")

        # Checkpoint every 5 batches
        if (i // batch_size) % 5 == 0:
            save_tensors(tensors_data)

        time.sleep(0.5)

    # Final save
    save_tensors(tensors_data)

    print(f"\n{'='*50}")
    print(f"  Download complete!")
    print(f"  New: {success_total}  |  Failed batches: {fail_total}")
    print(f"  Total tensors in JSON: {len(tensors_data)}")
    cif_count = len([f for f in os.listdir(DATASET_DIR) if f.endswith(".cif")])
    print(f"  Total CIF files: {cif_count}")
    print(f"{'='*50}")

    if fail_total > 0:
        print("  Re-run this script to retry failed materials.")


if __name__ == "__main__":
    fetch_all()
