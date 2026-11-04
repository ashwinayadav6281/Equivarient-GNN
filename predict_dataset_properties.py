"""
Unified property prediction for sAlex and OMat24 crystal datasets.

Predicts:
  1. Space Group (Number, International symbol, Hall symbol, Point group) via spglib
  2. Crystal System (Cubic, Tetragonal, Orthorhombic, Hexagonal, Trigonal, Monoclinic, Triclinic)
  3. Elastic Stiffness Tensor (6x6 matrix in GPa) and derived VRH Bulk & Shear Moduli via E3NN Equivariant GNN
  4. Electronic Band Gap (in eV) via pre-trained CGCNN model

Usage:
  python predict_dataset_properties.py --dataset salex --limit 5000 --output predictions_salex_5k.csv
  python predict_dataset_properties.py --dataset omat24 --subsplit rattled-relax --limit 10000 --output predictions_omat24_rattled_relax.csv
"""

import os
import sys
import math
import json
import csv
import time
import argparse
import tarfile
import lmdb
import zlib
import numpy as np
import torch
import spglib
from pymatgen.core import Structure

# Add project root and cgcnn module to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'cgcnn'))

from equivariant_model import SimpleEquivariantElasticNet
from tensor_utils import calculate_moduli
from cgcnn.model import CrystalGraphConvNet
from cgcnn.data import GaussianDistance, AtomCustomJSONInitializer

def parse_args():
    parser = argparse.ArgumentParser(description="Predict crystal properties on sAlex/OMat24")
    parser.add_argument("--dataset", type=str, default="salex", choices=["salex", "omat24", "custom"], help="Dataset to process")
    parser.add_argument("--subsplit", type=str, default="rattled-relax", help="Subsplit for OMat24 (e.g. rattled-relax, rattled-1000-subsampled)")
    parser.add_argument("--lmdb-path", type=str, default="", help="Custom path to an .aselmdb file if dataset=custom")
    parser.add_argument("--limit", type=int, default=0, help="Max structures to process (0 = all)")
    parser.add_argument("--output", type=str, default="", help="Output CSV path")
    parser.add_argument("--json-output", type=str, default="", help="Optional output JSON path")
    parser.add_argument("--print-freq", type=int, default=500, help="Progress logging frequency")
    return parser.parse_args()

def get_crystal_system(spg_num):
    if 1 <= spg_num <= 2: return "Triclinic"
    elif 3 <= spg_num <= 15: return "Monoclinic"
    elif 16 <= spg_num <= 74: return "Orthorhombic"
    elif 75 <= spg_num <= 142: return "Tetragonal"
    elif 143 <= spg_num <= 167: return "Trigonal"
    elif 168 <= spg_num <= 194: return "Hexagonal"
    elif 195 <= spg_num <= 230: return "Cubic"
    return "Unknown"

class UnifiedPredictor:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing UnifiedPredictor on {self.device}...")

        # 1. Load E3NN Elastic Tensor Model
        e3nn_ckpt_path = os.path.join(BASE_DIR, 'checkpoints_equivariant', 'model_best_equivariant.pth.tar')
        e3nn_stats_path = os.path.join(BASE_DIR, 'checkpoints_equivariant', 'normalization_stats.pt')

        stats = torch.load(e3nn_stats_path, map_location=self.device, weights_only=False)
        self.target_mean_np = stats['mean'].cpu().numpy()
        self.target_std_np = stats['std'].cpu().numpy()

        self.e3nn_model = SimpleEquivariantElasticNet(num_node_features=105).to(self.device)
        ckpt = torch.load(e3nn_ckpt_path, map_location=self.device, weights_only=False)
        self.e3nn_model.load_state_dict(ckpt['model_state_dict'])
        self.e3nn_model.eval()
        print("  [1/2] E3NN Elastic Tensor model loaded.")

        # 2. Load CGCNN Band Gap Model
        bg_ckpt_path = os.path.join(BASE_DIR, 'cgcnn', 'pre-trained', 'band-gap.pth.tar')
        atom_init_path = os.path.join(BASE_DIR, 'cgcnn', 'atom_init.json')

        bg_ckpt = torch.load(bg_ckpt_path, map_location=self.device, weights_only=False)
        bg_args = bg_ckpt['args']
        self.bg_mean = bg_ckpt['normalizer']['mean']
        self.bg_std = bg_ckpt['normalizer']['std']

        self.ari = AtomCustomJSONInitializer(atom_init_path)
        self.gdf = GaussianDistance(dmin=0, dmax=8.0, step=0.2)

        sample_fea = np.vstack([self.ari.get_atom_fea(14)])
        orig_atom_fea_len = sample_fea.shape[-1]
        nbr_fea_len = len(np.arange(0, 8.0 + 0.2, 0.2))

        self.bg_model = CrystalGraphConvNet(
            orig_atom_fea_len,
            nbr_fea_len,
            atom_fea_len=bg_args['atom_fea_len'],
            n_conv=bg_args['n_conv'],
            h_fea_len=bg_args['h_fea_len'],
            n_h=bg_args['n_h'],
            classification=False
        ).to(self.device)
        self.bg_model.load_state_dict(bg_ckpt['state_dict'])
        self.bg_model.eval()
        print("  [2/2] CGCNN Band Gap model loaded.")

    def predict(self, structure):
        # A. Space Group & Crystal System via spglib
        cell = (structure.lattice.matrix, structure.frac_coords, [site.specie.number for site in structure])
        sym = spglib.get_symmetry_dataset(cell, symprec=1e-2)
        if sym:
            spg_num = int(getattr(sym, 'number', sym['number']))
            spg_sym = str(getattr(sym, 'international', sym['international']))
            hall = str(getattr(sym, 'hall', sym['hall']))
            pt_grp = str(getattr(sym, 'pointgroup', sym['pointgroup']))
            crys_sys = get_crystal_system(spg_num)
        else:
            spg_num, spg_sym, hall, pt_grp, crys_sys = -1, "Unknown", "Unknown", "Unknown", "Unknown"

        # B. Elastic Tensors via E3NN
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
            rich_features.append([X / 4.0, radius / 3.0, mass / 200.0, group / 18.0, row / 7.0])

        one_hot = torch.nn.functional.one_hot(torch.tensor(atomic_numbers, dtype=torch.long), num_classes=100).to(torch.float32)
        rich_feat = torch.tensor(rich_features, dtype=torch.float32)
        node_features = torch.cat([one_hot, rich_feat], dim=-1).to(self.device)

        src_idx, dst_idx, edge_vecs = [], [], []
        for i in range(len(structure)):
            nbrs = structure.get_neighbors(structure[i], 6.0)
            for nbr in nbrs:
                src_idx.append(i)
                dst_idx.append(nbr.index)
                edge_vecs.append(nbr.coords - structure[i].coords)

        if src_idx:
            edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long).to(self.device)
            edge_vec = torch.tensor(np.array(edge_vecs), dtype=torch.float32).to(self.device)
            with torch.no_grad():
                pred_std = self.e3nn_model(node_features, edge_vec, edge_index).cpu().numpy()
                pred_C = pred_std * self.target_std_np + self.target_mean_np
                pred_C = (pred_C + pred_C.T) / 2.0
                moduli = calculate_moduli(pred_C)
                K_H = float(moduli.get('K_H', float('nan')))
                G_H = float(moduli.get('G_H', float('nan')))
                if math.isnan(K_H) or K_H < 0: K_H = 0.0
                if math.isnan(G_H) or G_H < 0: G_H = 0.0
                if K_H > 800: K_H = 800.0
                if G_H > 800: G_H = 800.0
        else:
            pred_C = np.zeros((6, 6))
            K_H, G_H = 0.0, 0.0

        # C. Band Gap via CGCNN
        atom_fea = np.vstack([self.ari.get_atom_fea(structure[i].specie.number) for i in range(len(structure))])
        all_nbrs = structure.get_all_neighbors(8.0, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < 12:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) + [0] * (12 - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) + [8.0 + 1.0] * (12 - len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:12])))
                nbr_fea.append(list(map(lambda x: x[1], nbr[:12])))

        nbr_fea = self.gdf.expand(np.array(nbr_fea))
        atom_fea_t = torch.Tensor(atom_fea).to(self.device)
        nbr_fea_t = torch.Tensor(nbr_fea).to(self.device)
        nbr_fea_idx_t = torch.LongTensor(nbr_fea_idx).to(self.device)
        crystal_atom_idx = [torch.LongTensor(list(range(len(structure)))).to(self.device)]

        with torch.no_grad():
            out_bg = self.bg_model(atom_fea_t, nbr_fea_t, nbr_fea_idx_t, crystal_atom_idx)
            pred_bg = max(0.0, float(out_bg.item() * self.bg_std + self.bg_mean))

        return {
            'space_group_number': spg_num,
            'space_group_symbol': spg_sym,
            'crystal_system': crys_sys,
            'hall_symbol': hall,
            'point_group': pt_grp,
            'predicted_band_gap_eV': round(pred_bg, 4),
            'predicted_bulk_modulus_VRH_GPa': round(K_H, 2),
            'predicted_shear_modulus_VRH_GPa': round(G_H, 2),
            'predicted_elastic_tensor_GPa': np.round(pred_C, 2).tolist()
        }

def find_aselmdb_files(dataset, subsplit, custom_path):
    if dataset == "custom":
        if os.path.exists(custom_path):
            return [custom_path]
        else:
            raise FileNotFoundError(f"Custom LMDB not found: {custom_path}")

    elif dataset == "salex":
        salex_dir = os.path.join(BASE_DIR, 'datasets', 'sAlex', 'train')
        if not os.path.exists(salex_dir):
            os.makedirs(salex_dir, exist_ok=True)
            tar_path = os.path.join(BASE_DIR, 'datasets', 'sAlex', 'train.tar.gz')
            print(f"Extracting sample from {tar_path}...")
            with tarfile.open(tar_path, 'r:gz') as tar:
                for member in tar.getmembers():
                    if member.name.endswith('.aselmdb'):
                        tar.extract(member, path=os.path.join(BASE_DIR, 'datasets', 'sAlex'))
                        break
        lmdbs = [os.path.join(salex_dir, f) for f in os.listdir(salex_dir) if f.endswith('.aselmdb')]
        if not lmdbs:
            # Extract at least one from tar.gz
            tar_path = os.path.join(BASE_DIR, 'datasets', 'sAlex', 'train.tar.gz')
            print(f"Extracting first LMDB from {tar_path}...")
            with tarfile.open(tar_path, 'r:gz') as tar:
                for member in tar.getmembers():
                    if member.name.endswith('.aselmdb'):
                        tar.extract(member, path=os.path.join(BASE_DIR, 'datasets', 'sAlex'))
                        lmdbs.append(os.path.join(BASE_DIR, 'datasets', 'sAlex', member.name))
                        break
        return lmdbs

    elif dataset == "omat24":
        omat_dir = os.path.join(BASE_DIR, 'datasets', 'omat24')
        extract_dir = os.path.join(omat_dir, subsplit)
        if not os.path.exists(extract_dir) or not any(f.endswith('.aselmdb') for f in os.listdir(extract_dir)):
            tar_path = os.path.join(omat_dir, f"{subsplit}.tar.gz")
            if not os.path.exists(tar_path):
                raise FileNotFoundError(f"Tarball not found: {tar_path}")
            print(f"Extracting LMDB files from {tar_path}...")
            os.makedirs(extract_dir, exist_ok=True)
            with tarfile.open(tar_path, 'r:gz') as tar:
                count = 0
                for member in tar.getmembers():
                    if member.name.endswith('.aselmdb'):
                        tar.extract(member, path=omat_dir)
                        count += 1
                        if count >= 3: # Extract first few LMDB databases for inference
                            break
        lmdbs = [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith('.aselmdb')]
        return lmdbs

    return []

def main():
    args = parse_args()
    lmdbs = find_aselmdb_files(args.dataset, args.subsplit, args.lmdb_path)
    print(f"Found {len(lmdbs)} LMDB databases to process:")
    for p in lmdbs:
        print(f"  - {p}")

    predictor = UnifiedPredictor()

    if not args.output:
        args.output = f"predictions_{args.dataset}_{args.subsplit if args.dataset == 'omat24' else 'train'}.csv"

    output_csv = os.path.join(BASE_DIR, args.output)
    print(f"\nWriting predictions to: {output_csv}")

    fieldnames = [
        'material_id', 'formula', 'num_atoms',
        'space_group_number', 'space_group_symbol', 'crystal_system', 'point_group',
        'predicted_band_gap_eV', 'predicted_bulk_modulus_VRH_GPa', 'predicted_shear_modulus_VRH_GPa',
        'elastic_tensor_C11', 'elastic_tensor_C12', 'elastic_tensor_C13',
        'elastic_tensor_C33', 'elastic_tensor_C44', 'elastic_tensor_C66',
        'full_elastic_tensor_json'
    ]

    all_results = []
    processed_count = 0
    start_time = time.time()

    with open(output_csv, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for lmdb_path in lmdbs:
            print(f"\nProcessing database: {lmdb_path}")
            env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    try:
                        decomp = zlib.decompress(value)
                        record = json.loads(decomp.decode('utf-8'))

                        cell = record['cell']
                        numbers = record['numbers']
                        positions = record['positions']
                        uid = record.get('unique_id', str(key))
                        sid = record.get('data', {}).get('sid', str(uid))

                        struct = Structure(lattice=cell, species=numbers, coords=positions, coords_are_cartesian=True)
                        preds = predictor.predict(struct)

                        C = preds['predicted_elastic_tensor_GPa']

                        row = {
                            'material_id': sid,
                            'formula': struct.composition.reduced_formula,
                            'num_atoms': len(struct),
                            'space_group_number': preds['space_group_number'],
                            'space_group_symbol': preds['space_group_symbol'],
                            'crystal_system': preds['crystal_system'],
                            'point_group': preds['point_group'],
                            'predicted_band_gap_eV': preds['predicted_band_gap_eV'],
                            'predicted_bulk_modulus_VRH_GPa': preds['predicted_bulk_modulus_VRH_GPa'],
                            'predicted_shear_modulus_VRH_GPa': preds['predicted_shear_modulus_VRH_GPa'],
                            'elastic_tensor_C11': C[0][0],
                            'elastic_tensor_C12': C[0][1],
                            'elastic_tensor_C13': C[0][2],
                            'elastic_tensor_C33': C[2][2],
                            'elastic_tensor_C44': C[3][3],
                            'elastic_tensor_C66': C[5][5],
                            'full_elastic_tensor_json': json.dumps(C)
                        }

                        writer.writerow(row)
                        csv_file.flush()

                        if args.json_output:
                            all_results.append(row)

                        processed_count += 1
                        if processed_count % args.print_freq == 0:
                            elapsed = time.time() - start_time
                            rate = processed_count / max(elapsed, 1e-3)
                            print(f"  [{processed_count}] {rate:.1f} crystals/s | Last: {struct.composition.reduced_formula} (Spg: {preds['space_group_symbol']}, Eg: {preds['predicted_band_gap_eV']} eV, K: {preds['predicted_bulk_modulus_VRH_GPa']} GPa, G: {preds['predicted_shear_modulus_VRH_GPa']} GPa)")

                        if args.limit > 0 and processed_count >= args.limit:
                            print(f"\nReached requested limit of {args.limit} structures.")
                            break

                    except Exception as e:
                        continue

            if args.limit > 0 and processed_count >= args.limit:
                break

    if args.json_output:
        json_path = os.path.join(BASE_DIR, args.json_output)
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(all_results, jf, indent=2)
        print(f"Saved JSON results to {json_path}")

    elapsed = time.time() - start_time
    print(f"\n=== PREDICTION COMPLETE ===")
    print(f"Total processed: {processed_count} materials in {elapsed:.1f}s ({processed_count / max(elapsed, 1e-3):.1f} crystals/s)")
    print(f"Output saved to: {output_csv}")

if __name__ == '__main__':
    main()
