import os
import json
import glob
import time
from collections import defaultdict
import multiprocessing
from pymatgen.core import Structure
import csv
import shutil

DATA_DIR = '/home/ashwinay23/UGP-Phy'
DATASET_DIR = os.path.join(DATA_DIR, 'dataset_equivariant')
TENSOR_JSON = os.path.join(DATASET_DIR, 'elastic_tensors.json')

def process_cif(filepath):
    """Parse a single CIF file and return its properties."""
    try:
        mat_id = os.path.basename(filepath).replace('.cif', '')
        structure = Structure.from_file(filepath)
        
        formula = structure.composition.reduced_formula
        # Suppress symmetry warnings
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sg_info = structure.get_space_group_info(symprec=0.1)
        sg_num = sg_info[1]
        
        vol_per_atom = structure.volume / len(structure)
        
        return {
            'mat_id': mat_id,
            'formula': formula,
            'sg': sg_num,
            'vol_per_atom': vol_per_atom,
            'filepath': filepath
        }
    except Exception as e:
        return {'mat_id': os.path.basename(filepath).replace('.cif', ''), 'error': str(e)}

def main():
    print("Finding CIF files...")
    cif_files = glob.glob(os.path.join(DATASET_DIR, '*.cif'))
    print(f"Found {len(cif_files)} CIF files.")
    
    # 1. Parse all structures in parallel
    print("Parsing structures (this may take a minute)...")
    start_time = time.time()
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(process_cif, cif_files)
    print(f"Parsing finished in {time.time() - start_time:.1f} seconds.")
    
    # Filter out errors
    valid_mats = [r for r in results if 'error' not in r]
    print(f"Successfully parsed {len(valid_mats)} structures.")
    
    # 2. Group by (Formula, Space Group)
    groups = defaultdict(list)
    for m in valid_mats:
        key = (m['formula'], m['sg'])
        groups[key].append(m)
        
    # 3. Find duplicates based on Volume per atom (1.5% tolerance)
    to_keep = []
    to_drop = []
    
    VOL_TOLERANCE = 0.015  # 1.5% tolerance
    
    for key, mats in groups.items():
        if len(mats) == 1:
            to_keep.append(mats[0]['mat_id'])
            continue
            
        # Sort by volume
        mats.sort(key=lambda x: x['vol_per_atom'])
        
        # Group by similar volume
        clusters = []
        for m in mats:
            added = False
            for cluster in clusters:
                # Compare to the first item in the cluster
                ref_vol = cluster[0]['vol_per_atom']
                if abs(m['vol_per_atom'] - ref_vol) / ref_vol <= VOL_TOLERANCE:
                    cluster.append(m)
                    added = True
                    break
            if not added:
                clusters.append([m])
                
        # Resolve clusters
        for cluster in clusters:
            if len(cluster) == 1:
                to_keep.append(cluster[0]['mat_id'])
            else:
                # We have duplicates! Prefer MP over JARVIS
                cluster.sort(key=lambda x: 0 if x['mat_id'].startswith('mp-') else 1)
                
                keeper = cluster[0]
                to_keep.append(keeper['mat_id'])
                
                # The rest are duplicates to drop
                for duplicate in cluster[1:]:
                    to_drop.append(duplicate['mat_id'])
                    
    print(f"\nDuplicate Resolution:")
    print(f"  Materials to keep: {len(to_keep)}")
    print(f"  Materials to drop (duplicates): {len(to_drop)}")
    
    if len(to_drop) == 0:
        print("No duplicates found. Exiting.")
        return
        
    # 4. Backup the dataset folder just in case (optional, but requested to overwrite, so we will just overwrite directly)
    # The user said "do not create new datasets just edit the inital ones"
    
    # 5. Delete duplicate CIF files
    print("\nDeleting duplicate CIF files...")
    deleted_count = 0
    for mat_id in to_drop:
        filepath = os.path.join(DATASET_DIR, f"{mat_id}.cif")
        if os.path.exists(filepath):
            os.remove(filepath)
            deleted_count += 1
    print(f"Deleted {deleted_count} CIF files.")
    
    # 6. Update elastic_tensors.json
    print("Updating elastic_tensors.json...")
    with open(TENSOR_JSON, 'r') as f:
        tensors = json.load(f)
        
    orig_tensor_count = len(tensors)
    for mat_id in to_drop:
        tensors.pop(mat_id, None)
        
    with open(TENSOR_JSON, 'w') as f:
        json.dump(tensors, f)
    print(f"Tensors JSON updated: {orig_tensor_count} -> {len(tensors)} entries.")
    
    # 7. Update bandgaps_all.csv if it exists
    bg_csv = os.path.join(DATA_DIR, 'bandgaps_all.csv')
    if os.path.exists(bg_csv):
        print("Updating bandgaps_all.csv...")
        kept_rows = []
        with open(bg_csv, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            kept_rows.append(header)
            for row in reader:
                if row[0] not in to_drop:
                    kept_rows.append(row)
        with open(bg_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(kept_rows)
        print(f"Bandgaps CSV updated: kept {len(kept_rows)-1} entries.")

    print("\nDataset successfully deduplicated!")

if __name__ == '__main__':
    # Add project protection for multiprocessing on Windows
    multiprocessing.freeze_support()
    main()
