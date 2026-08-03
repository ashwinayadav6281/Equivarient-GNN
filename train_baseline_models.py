"""
Train Linear Regression and Random Forest models for Bulk & Shear Modulus
prediction, and compare with CGCNN results.

Usage:
    python train_baseline_models.py

Outputs:
    - features_cache.csv          (cached extracted features)
    - predictions_lr_bulk.csv     (Linear Regression bulk predictions)
    - predictions_lr_shear.csv    (Linear Regression shear predictions)
    - predictions_rf_bulk.csv     (Random Forest bulk predictions)
    - predictions_rf_shear.csv    (Random Forest shear predictions)
    - Final comparison table printed to console
"""

import csv
import os
import sys
import time
import warnings
import random

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = "cgcnn_dataset_full"
FEATURES_CACHE = "features_cache.csv"
RANDOM_SEED = 123
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# ============================================================
# FEATURE EXTRACTION
# ============================================================

# Elemental properties to aggregate across composition
ELEM_PROPS = [
    'atomic_mass', 'atomic_radius', 'X',  # X = Pauling electronegativity
    'group', 'row', 'mendeleev_no',
]


def safe_get_prop(element, prop_name):
    """Safely get an elemental property, returning NaN if unavailable."""
    try:
        val = getattr(element, prop_name)
        if val is None:
            return float('nan')
        return float(val)
    except (AttributeError, TypeError):
        return float('nan')


def extract_features_single(structure):
    """
    Extract composition + structural features from a pymatgen Structure.
    Returns a dict of feature_name -> value.
    """
    features = {}

    # --- Structural features ---
    features['volume'] = structure.volume
    features['volume_per_atom'] = structure.volume / len(structure)
    features['density'] = structure.density
    features['num_sites'] = len(structure)
    features['num_elements'] = len(set(structure.species))

    # Lattice parameters
    lattice = structure.lattice
    features['lattice_a'] = lattice.a
    features['lattice_b'] = lattice.b
    features['lattice_c'] = lattice.c
    features['lattice_alpha'] = lattice.alpha
    features['lattice_beta'] = lattice.beta
    features['lattice_gamma'] = lattice.gamma

    # Lattice ratios
    features['lattice_b_over_a'] = lattice.b / lattice.a if lattice.a > 0 else 0
    features['lattice_c_over_a'] = lattice.c / lattice.a if lattice.a > 0 else 0

    # --- Composition features ---
    # Get element fractions
    composition = structure.composition
    elements = composition.elements
    fractions = [composition.get_atomic_fraction(el) for el in elements]

    for prop_name in ELEM_PROPS:
        values = [safe_get_prop(el, prop_name) for el in elements]

        # Filter out NaN values
        valid_pairs = [(v, f) for v, f in zip(values, fractions) if not np.isnan(v)]
        if not valid_pairs:
            features[f'{prop_name}_mean'] = 0.0
            features[f'{prop_name}_std'] = 0.0
            features[f'{prop_name}_min'] = 0.0
            features[f'{prop_name}_max'] = 0.0
            features[f'{prop_name}_range'] = 0.0
            features[f'{prop_name}_wmean'] = 0.0
            continue

        vals = [v for v, _ in valid_pairs]
        fracs = [f for _, f in valid_pairs]

        features[f'{prop_name}_mean'] = np.mean(vals)
        features[f'{prop_name}_std'] = np.std(vals) if len(vals) > 1 else 0.0
        features[f'{prop_name}_min'] = np.min(vals)
        features[f'{prop_name}_max'] = np.max(vals)
        features[f'{prop_name}_range'] = np.max(vals) - np.min(vals)
        # Weighted mean by composition fraction
        features[f'{prop_name}_wmean'] = sum(v * f for v, f in zip(vals, fracs))

    return features


def extract_all_features(data_dir, cif_ids, cache_path=FEATURES_CACHE):
    """
    Extract features for all CIF files. Uses cache if available.
    Returns (feature_names, feature_matrix) where feature_matrix is np.array of shape (N, D).
    """
    from pymatgen.core.structure import Structure

    # Check cache
    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        with open(cache_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            feat_names = header[1:]  # skip 'cif_id'

            cached = {}
            for row in reader:
                if not row:
                    continue
                cached[row[0]] = [float(x) for x in row[1:]]

        # Check if all IDs are cached
        missing = [cid for cid in cif_ids if cid not in cached]
        if not missing:
            print(f"  All {len(cif_ids)} structures found in cache.")
            X = np.array([cached[cid] for cid in cif_ids])
            return feat_names, X
        else:
            print(f"  Cache missing {len(missing)} structures. Re-extracting all...")

    # Extract features
    print(f"Extracting features from {len(cif_ids)} CIF files...")
    print("  This may take 30-60 minutes for 13k structures. Progress shown below.")

    all_features = {}
    feat_names = None
    failed = []
    start_time = time.time()

    for i, cif_id in enumerate(cif_ids):
        if (i + 1) % 500 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(cif_ids) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(cif_ids)}] {rate:.1f} structs/sec, "
                  f"ETA: {eta/60:.1f} min")

        cif_path = os.path.join(data_dir, cif_id + '.cif')
        try:
            structure = Structure.from_file(cif_path)
            feats = extract_features_single(structure)
            if feat_names is None:
                feat_names = sorted(feats.keys())
            all_features[cif_id] = [feats[k] for k in feat_names]
        except Exception as e:
            failed.append(cif_id)

    elapsed = time.time() - start_time
    print(f"  Done in {elapsed/60:.1f} min. Extracted: {len(all_features)}, "
          f"Failed: {len(failed)}")

    # Save cache
    print(f"  Saving cache to {cache_path}...")
    with open(cache_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['cif_id'] + feat_names)
        for cid in sorted(all_features.keys()):
            writer.writerow([cid] + all_features[cid])

    # Build feature matrix for requested IDs
    X = []
    valid_ids = []
    for cid in cif_ids:
        if cid in all_features:
            X.append(all_features[cid])
            valid_ids.append(cid)

    return feat_names, np.array(X)


# ============================================================
# DATA LOADING & FILTERING
# ============================================================

def load_targets(csv_path):
    """Load id_prop CSV and return list of (cif_id, target_value)."""
    data = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            data.append((row[0], float(row[1])))
    return data


def filter_missing_cifs(data, data_dir):
    """Remove entries whose CIF files don't exist."""
    before = len(data)
    data = [(cid, t) for cid, t in data
            if os.path.exists(os.path.join(data_dir, cid + '.cif'))]
    skipped = before - len(data)
    if skipped > 0:
        print(f"  Filtered {skipped} entries with missing CIF files.")
    return data


def filter_outliers_iqr(data):
    """Remove extreme outlier targets using IQR fences (same as CGCNN data.py)."""
    targets = [t for _, t in data]
    sorted_t = sorted(targets)
    n = len(sorted_t)
    q1 = sorted_t[n // 4]
    q3 = sorted_t[3 * n // 4]
    iqr = q3 - q1
    lower = q1 - 3.0 * iqr
    upper = q3 + 3.0 * iqr

    before = len(data)
    data = [(cid, t) for cid, t in data if lower <= t <= upper]
    removed = before - len(data)
    if removed > 0:
        print(f"  Removed {removed} outliers outside [{lower:.1f}, {upper:.1f}].")
    return data


def split_data(data, train_ratio, val_ratio, test_ratio, seed):
    """Split into train/val/test using the same logic as CGCNN."""
    random.seed(seed)
    indices = list(range(len(data)))
    random.shuffle(indices)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return train_idx, val_idx, test_idx


# ============================================================
# TRAINING & EVALUATION
# ============================================================

def train_and_evaluate(X, y, cif_ids, train_idx, val_idx, test_idx,
                       model_name, target_name, output_csv):
    """Train a model, evaluate on test set, save predictions."""
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Replace NaN/inf with 0 (safety)
    X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_s = np.nan_to_num(X_val_s, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_s = np.nan_to_num(X_test_s, nan=0.0, posinf=0.0, neginf=0.0)

    # Create model
    if model_name == 'Linear Regression':
        model = LinearRegression()
    elif model_name == 'Random Forest':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Train
    print(f"    Training {model_name}...")
    start = time.time()
    model.fit(X_train_s, y_train)
    train_time = time.time() - start
    print(f"    Training done in {train_time:.1f}s")

    # Predict on test set
    y_pred_test = model.predict(X_test_s)

    # Metrics on test set
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)

    # Also compute train MAE for reference
    y_pred_train = model.predict(X_train_s)
    train_mae = mean_absolute_error(y_train, y_pred_train)

    print(f"    Train MAE: {train_mae:.3f} GPa")
    print(f"    Test  MAE: {mae:.3f} GPa | RMSE: {rmse:.3f} GPa | R2: {r2:.4f}")

    # Save test predictions
    test_cif_ids = [cif_ids[i] for i in test_idx]
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['cif_id', 'actual', 'predicted'])
        for cid, actual, pred in zip(test_cif_ids, y_test, y_pred_test):
            writer.writerow([cid, f'{actual:.4f}', f'{pred:.4f}'])
    print(f"    Predictions saved to {output_csv}")

    # Feature importance for Random Forest
    if model_name == 'Random Forest':
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[-10:][::-1]
        print(f"    Top 10 features:")
        for idx in top_idx:
            print(f"      {feat_names[idx]:30s}  {importances[idx]:.4f}")

    return {
        'model': model_name,
        'target': target_name,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'train_mae': train_mae,
        'train_time': train_time,
        'n_train': len(train_idx),
        'n_test': len(test_idx),
    }


# ============================================================
# LOAD CGCNN RESULTS FOR COMPARISON
# ============================================================

def load_cgcnn_metrics(predictions_csv, target_name):
    """Load CGCNN predictions and compute metrics on the same basis."""
    if not os.path.exists(predictions_csv):
        print(f"  WARNING: {predictions_csv} not found. Skipping CGCNN comparison.")
        return None

    actuals, preds = [], []
    with open(predictions_csv) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # Skip header if present
            try:
                actuals.append(float(row[1]))
                preds.append(float(row[2]))
            except (ValueError, IndexError):
                continue

    if not actuals:
        return None

    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    r2 = r2_score(actuals, preds)

    return {
        'model': 'CGCNN',
        'target': target_name,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'train_mae': '-',
        'train_time': '-',
        'n_train': '-',
        'n_test': len(actuals),
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    print("=" * 60)
    print("  BASELINE MODEL TRAINING: Linear Regression & Random Forest")
    print("=" * 60)

    # --- Load both target CSVs ---
    print("\n[1/5] Loading target data...")
    bulk_data = load_targets(os.path.join(DATA_DIR, "id_prop_bulk.csv"))
    shear_data = load_targets(os.path.join(DATA_DIR, "id_prop_shear.csv"))
    print(f"  Bulk entries: {len(bulk_data)}")
    print(f"  Shear entries: {len(shear_data)}")

    # --- Filter missing CIFs ---
    print("\n[2/5] Filtering missing CIFs and outliers...")
    print("  Bulk:")
    bulk_data = filter_missing_cifs(bulk_data, DATA_DIR)
    bulk_data = filter_outliers_iqr(bulk_data)
    print("  Shear:")
    shear_data = filter_missing_cifs(shear_data, DATA_DIR)
    shear_data = filter_outliers_iqr(shear_data)

    # Get union of all valid CIF IDs
    bulk_ids = {cid for cid, _ in bulk_data}
    shear_ids = {cid for cid, _ in shear_data}
    all_ids = sorted(bulk_ids | shear_ids)
    print(f"\n  Total unique structures: {len(all_ids)}")

    # --- Extract features ---
    print(f"\n[3/5] Feature extraction...")
    feat_names, X_all = extract_all_features(DATA_DIR, all_ids)
    print(f"  Feature matrix shape: {X_all.shape}")
    print(f"  Features ({len(feat_names)}): {feat_names[:5]}... {feat_names[-5:]}")

    # Build ID->index mapping
    id_to_idx = {cid: i for i, cid in enumerate(all_ids)}

    # --- Train models for each target ---
    all_results = []

    for target_name, target_data, cgcnn_csv in [
        ('Bulk Modulus', bulk_data, 'predictions_bulk.csv'),
        ('Shear Modulus', shear_data, 'predictions_shear.csv'),
    ]:
        print(f"\n{'='*60}")
        print(f"  TARGET: {target_name}")
        print(f"{'='*60}")

        # Build aligned X, y arrays
        cif_ids = [cid for cid, _ in target_data]
        y = np.array([t for _, t in target_data])
        indices = [id_to_idx[cid] for cid in cif_ids]
        X = X_all[indices]

        print(f"  Samples: {len(cif_ids)}, Features: {X.shape[1]}")
        print(f"  Target range: [{y.min():.2f}, {y.max():.2f}], mean: {y.mean():.2f}")

        # Split
        print(f"\n[4/5] Splitting data (80/10/10)...")
        train_idx, val_idx, test_idx = split_data(
            target_data, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
        )
        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

        # Train Linear Regression
        short_target = target_name.split()[0].lower()  # 'bulk' or 'shear'
        print(f"\n  --- Linear Regression ---")
        lr_result = train_and_evaluate(
            X, y, cif_ids, train_idx, val_idx, test_idx,
            'Linear Regression', target_name,
            f'predictions_lr_{short_target}.csv'
        )
        all_results.append(lr_result)

        # Train Random Forest
        print(f"\n  --- Random Forest ---")
        rf_result = train_and_evaluate(
            X, y, cif_ids, train_idx, val_idx, test_idx,
            'Random Forest', target_name,
            f'predictions_rf_{short_target}.csv'
        )
        all_results.append(rf_result)

        # Load CGCNN results
        cgcnn_result = load_cgcnn_metrics(cgcnn_csv, target_name)
        if cgcnn_result:
            all_results.append(cgcnn_result)

    # --- Print comparison table ---
    print(f"\n\n{'='*80}")
    print("  FINAL MODEL COMPARISON")
    print(f"{'='*80}\n")

    header = f"{'Model':<22} {'Target':<16} {'MAE (GPa)':>10} {'RMSE (GPa)':>11} {'R2':>8} {'Train Time':>12}"
    print(header)
    print("-" * len(header))

    # Group by target
    for target in ['Bulk Modulus', 'Shear Modulus']:
        target_results = [r for r in all_results if r['target'] == target]
        # Sort: CGCNN first, then LR, then RF
        order = {'CGCNN': 0, 'Linear Regression': 1, 'Random Forest': 2}
        target_results.sort(key=lambda r: order.get(r['model'], 99))

        for r in target_results:
            tt = f"{r['train_time']:.1f}s" if isinstance(r['train_time'], float) else r['train_time']
            print(f"{r['model']:<22} {r['target']:<16} {r['mae']:>10.3f} {r['rmse']:>11.3f} {r['r2']:>8.4f} {tt:>12}")
        print()

    print("=" * 80)
    print("  DONE! Prediction files saved.")
    print("=" * 80)
