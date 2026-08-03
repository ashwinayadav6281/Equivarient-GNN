"""
Predict bulk and shear modulus for all crystals in cgcnn_dataset_full
using the trained CGCNN models.

Outputs:
  - predictions_bulk.csv    (cif_id, actual, predicted)
  - predictions_shear.csv   (cif_id, actual, predicted)
  - predictions_combined.csv (cif_id, bulk_actual, bulk_predicted, shear_actual, shear_predicted)
"""
import os
import sys
import csv
import shutil
import subprocess


DATA_DIR = "cgcnn_dataset_full"
PYTHON = sys.executable

MODELS = {
    "bulk": {
        "model": "model_best_bulk.pth.tar",
        "source_csv": os.path.join(DATA_DIR, "id_prop_bulk.csv"),
    },
    "shear": {
        "model": "model_best_shear.pth.tar",
        "source_csv": os.path.join(DATA_DIR, "id_prop_shear.csv"),
    },
}


def run_prediction(target_name):
    config = MODELS[target_name]
    model_path = config["model"]
    source_csv = config["source_csv"]
    active_csv = os.path.join(DATA_DIR, "id_prop.csv")
    output_csv = f"predictions_{target_name}.csv"

    print(f"\n{'='*55}")
    print(f"  PREDICTING: {target_name.upper()} MODULUS")
    print(f"{'='*55}")

    # Check model exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        return None

    # Copy the correct target CSV as id_prop.csv
    shutil.copyfile(source_csv, active_csv)
    print(f"  Dataset:  {source_csv}")
    print(f"  Model:    {model_path}")

    # Run CGCNN predict.py
    cmd = [
        PYTHON,
        os.path.join("cgcnn", "predict.py"),
        model_path,
        DATA_DIR,
        "--batch-size", "256",
        "--workers", "0",
    ]
    print(f"  Command:  {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"  Prediction FAILED for {target_name}!")
        return None

    # Rename test_results.csv to predictions_{target}.csv
    if os.path.exists("test_results.csv"):
        os.replace("test_results.csv", output_csv)
        print(f"\n  Saved: {output_csv}")
    else:
        print("  WARNING: test_results.csv not generated")
        return None

    # Read and summarize
    results = {}
    with open(output_csv) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            cif_id, actual, predicted = row[0], float(row[1]), float(row[2])
            results[cif_id] = (actual, predicted)

    errors = [abs(a - p) for a, p in results.values()]
    mae = sum(errors) / len(errors)
    print(f"  Total predictions: {len(results)}")
    print(f"  MAE: {mae:.3f} GPa")

    return results


def combine_results(bulk_results, shear_results, output_path="predictions_combined.csv"):
    """Combine bulk and shear predictions into a single CSV."""
    # Get all CIF IDs that appear in both
    all_ids = set(bulk_results.keys()) | set(shear_results.keys())
    all_ids = sorted(all_ids)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "cif_id",
            "bulk_actual", "bulk_predicted", "bulk_error",
            "shear_actual", "shear_predicted", "shear_error"
        ])
        for cif_id in all_ids:
            bulk_a, bulk_p, bulk_e = "", "", ""
            shear_a, shear_p, shear_e = "", "", ""

            if cif_id in bulk_results:
                bulk_a, bulk_p = bulk_results[cif_id]
                bulk_e = abs(bulk_a - bulk_p)
            if cif_id in shear_results:
                shear_a, shear_p = shear_results[cif_id]
                shear_e = abs(shear_a - shear_p)

            writer.writerow([
                cif_id,
                f"{bulk_a:.3f}" if bulk_a != "" else "",
                f"{bulk_p:.3f}" if bulk_p != "" else "",
                f"{bulk_e:.3f}" if bulk_e != "" else "",
                f"{shear_a:.3f}" if shear_a != "" else "",
                f"{shear_p:.3f}" if shear_p != "" else "",
                f"{shear_e:.3f}" if shear_e != "" else "",
            ])

    print(f"\nCombined results saved to: {output_path}")
    print(f"Total entries: {len(all_ids)}")


if __name__ == "__main__":
    # Run predictions for both properties
    bulk_results = run_prediction("bulk")
    shear_results = run_prediction("shear")

    # Combine into a single file
    if bulk_results and shear_results:
        combine_results(bulk_results, shear_results)

    # Cleanup
    active_csv = os.path.join(DATA_DIR, "id_prop.csv")
    if os.path.exists(active_csv):
        os.remove(active_csv)

    print(f"\n{'='*55}")
    print("  ALL PREDICTIONS COMPLETE!")
    print(f"{'='*55}")
    print("\nOutput files:")
    for f in ["predictions_bulk.csv", "predictions_shear.csv", "predictions_combined.csv"]:
        if os.path.exists(f):
            print(f"  - {f}")
