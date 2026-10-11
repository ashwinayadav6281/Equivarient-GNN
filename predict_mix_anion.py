"""
Predict elastic tensors for mix-anion compounds using the trained E3NN model.
Reads VASP/POSCAR structures, runs inference, and writes results back to the JSON.
"""
import os
import sys
import json
import math
import glob
import torch
import numpy as np
from pymatgen.core import Structure
import warnings
warnings.filterwarnings('ignore')

# Add project root to path (works on both Windows and Linux)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from equivariant_model import SimpleEquivariantElasticNet
from tensor_utils import calculate_moduli

def process_vasp_structure(vasp_path, max_radius=6.0):
    """Parse a VASP/POSCAR file and build the graph with 105-dim rich features."""
    try:
        structure = Structure.from_file(vasp_path)
    except Exception as e:
        print(f"  ERROR parsing {vasp_path}: {e}")
        return None

    atomic_numbers = []
    rich_features = []
    for site in structure:
        sp = site.specie
        num = min(sp.number, 99)
        atomic_numbers.append(num)

        X = sp.X if sp.X is not None and not math.isnan(sp.X) else 0.0
        radius = sp.atomic_radius if sp.atomic_radius is not None and not (isinstance(sp.atomic_radius, float) and math.isnan(sp.atomic_radius)) else 0.0
        mass = sp.atomic_mass if sp.atomic_mass is not None and not (isinstance(sp.atomic_mass, float) and math.isnan(sp.atomic_mass)) else 0.0
        group = sp.group if sp.group is not None else 0.0
        row = sp.row if sp.row is not None else 0.0

        rich_features.append([
            X / 4.0,
            radius / 3.0,
            mass / 200.0,
            group / 18.0,
            row / 7.0
        ])

    atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long)
    one_hot = torch.nn.functional.one_hot(atomic_numbers, num_classes=100).to(torch.float32)
    rich_feat = torch.tensor(rich_features, dtype=torch.float32)
    node_features = torch.cat([one_hot, rich_feat], dim=-1)

    src_indices, dst_indices, edge_vectors = [], [], []
    N = len(structure)
    for i in range(N):
        neighbors = structure.get_neighbors(structure[i], max_radius)
        for neighbor in neighbors:
            j = neighbor.index
            vec = neighbor.coords - structure[i].coords
            src_indices.append(i)
            dst_indices.append(j)
            edge_vectors.append(vec)

    if not src_indices:
        return None

    edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
    edge_vec = torch.tensor(np.array(edge_vectors), dtype=torch.float32)

    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_vec': edge_vec,
        'formula': structure.composition.reduced_formula,
        'num_atoms': N,
    }


def main():
    DATA_DIR = os.environ.get('UGP_DIR', os.path.dirname(os.path.abspath(__file__)))
    # If running from project root, use current dir
    if not os.path.exists(os.path.join(DATA_DIR, 'mix_anion')):
        DATA_DIR = os.getcwd()
    MIX_DIR = os.path.join(DATA_DIR, 'mix_anion')
    STRUCT_DIR = os.path.join(MIX_DIR, 'parent_structures')
    CKPT_DIR = os.path.join(DATA_DIR, 'checkpoints_equivariant')

    device = torch.device('cpu')

    # 1. Load model
    print("Loading E3NN model (105-dim features)...")
    model = SimpleEquivariantElasticNet(num_node_features=105).to(device)
    ckpt = torch.load(os.path.join(CKPT_DIR, 'model_best_equivariant.pth.tar'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Model loaded (best val_loss: {ckpt.get('best_val_loss', 'N/A')})")

    # 2. Load normalization stats
    stats = torch.load(os.path.join(CKPT_DIR, 'normalization_stats.pt'), map_location=device)
    target_mean = stats['mean'].cpu().numpy()
    target_std = stats['std'].cpu().numpy()

    # 3. Load existing JSON
    json_path = os.path.join(MIX_DIR, 'standard_wyckoff_complete_weighted.json')
    with open(json_path, 'r') as f:
        compounds = json.load(f)
    print(f"Loaded {len(compounds)} mix-anion compounds from JSON.\n")

    # 4. Build a lookup from OQMD_ID -> VASP file path
    parent_files = glob.glob(os.path.join(MIX_DIR, 'parent_structures', '*.vasp'))
    child_files = glob.glob(os.path.join(MIX_DIR, 'child_structures', '*.vasp'))
    vasp_files = parent_files + child_files
    
    vasp_lookup = {}
    for vf in vasp_files:
        # Filename: OQMD_id_12345.vasp
        basename = os.path.basename(vf).replace('.vasp', '')
        vasp_lookup[basename] = vf
    print(f"Found {len(vasp_lookup)} VASP structure files.\n")

    # 5. For each compound in JSON, predict elastic tensor
    # The compounds reference parent structures via parent_compounds[].OQMD_ID (integer)
    # But we also want to predict for the mix-anion compound itself if it has a structure
    
    # First, collect all unique OQMD IDs that have structures
    all_predictions = {}  # OQMD_id_XXXXX -> prediction dict
    
    # Collect all OQMD IDs we need predictions for
    oqmd_ids_needed = set()
    for comp in compounds:
        oqmd_id = comp['OQMD_ID']  # e.g., "OQMD_id_1297130"
        if oqmd_id in vasp_lookup:
            oqmd_ids_needed.add(oqmd_id)
        # Also check parent compounds
        for parent in comp.get('parent_compounds', []):
            parent_id = f"OQMD_id_{parent['OQMD_ID']}"
            if parent_id in vasp_lookup:
                oqmd_ids_needed.add(parent_id)
    
    print(f"Need predictions for {len(oqmd_ids_needed)} unique structures.\n")
    
    # Run inference on all unique structures
    success = 0
    fail = 0
    
    with torch.no_grad():
        for oqmd_id in sorted(oqmd_ids_needed):
            vasp_path = vasp_lookup[oqmd_id]
            graph = process_vasp_structure(vasp_path)
            if graph is None:
                print(f"  SKIP {oqmd_id}: could not parse structure")
                fail += 1
                continue

            node_features = graph['node_features'].to(device)
            edge_vec = graph['edge_vec'].to(device)
            edge_index = graph['edge_index'].to(device)

            pred_std = model(node_features, edge_vec, edge_index).cpu().numpy()
            C = pred_std * target_std + target_mean

            # Force symmetry
            C = (C + C.T) / 2.0

            try:
                moduli = calculate_moduli(C)
                k_vrh = float(moduli.get('K_H', 0))
                g_vrh = float(moduli.get('G_H', 0))
            except:
                k_vrh = 0.0
                g_vrh = 0.0

            all_predictions[oqmd_id] = {
                'elastic_tensor_GPa': C.tolist(),
                'bulk_modulus_VRH_GPa': round(k_vrh, 2),
                'shear_modulus_VRH_GPa': round(g_vrh, 2),
                'formula': graph['formula'],
                'num_atoms': graph['num_atoms'],
            }
            success += 1
            print(f"  {oqmd_id} ({graph['formula']}): K={k_vrh:.1f} GPa, G={g_vrh:.1f} GPa")

    print(f"\nInference complete: {success} success, {fail} failed.\n")

    # 6. Write predictions back into JSON
    for comp in compounds:
        oqmd_id = comp['OQMD_ID']
        if oqmd_id in all_predictions:
            pred = all_predictions[oqmd_id]
            comp['predicted_elastic_tensor_GPa'] = pred['elastic_tensor_GPa']
            comp['predicted_bulk_modulus_VRH_GPa'] = pred['bulk_modulus_VRH_GPa']
            comp['predicted_shear_modulus_VRH_GPa'] = pred['shear_modulus_VRH_GPa']

        # Also enrich parent compounds
        for parent in comp.get('parent_compounds', []):
            parent_key = f"OQMD_id_{parent['OQMD_ID']}"
            if parent_key in all_predictions:
                pred = all_predictions[parent_key]
                parent['predicted_elastic_tensor_GPa'] = pred['elastic_tensor_GPa']
                parent['predicted_bulk_modulus_VRH_GPa'] = pred['bulk_modulus_VRH_GPa']
                parent['predicted_shear_modulus_VRH_GPa'] = pred['shear_modulus_VRH_GPa']

    # 7. Save updated JSON
    output_path = os.path.join(MIX_DIR, 'standard_wyckoff_complete_weighted.json')
    with open(output_path, 'w') as f:
        json.dump(compounds, f, indent=2)
    print(f"Updated JSON saved to: {output_path}")
    print("Done!")


if __name__ == '__main__':
    main()
