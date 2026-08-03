import os
import shutil
import subprocess
import sys
import pandas as pd


def predict_new_structures(cif_directory, checkpoint_path="model_best.pth.tar"):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found.")

    predict_dir = "cgcnn_predict_temp"
    os.makedirs(predict_dir, exist_ok=True)

    # Copy atom feature definition
    shutil.copyfile(
        os.path.join("cgcnn_dataset", "atom_init.json"),
        os.path.join(predict_dir, "atom_init.json"),
    )

    # Gather target CIF files and build dummy id_prop.csv
    cif_files = [f for f in os.listdir(cif_directory) if f.endswith(".cif")]
    if not cif_files:
        print(f"No .cif files found in '{cif_directory}'.")
        return

    dummy_records = []
    for fname in cif_files:
        mat_id = os.path.splitext(fname)[0]
        shutil.copyfile(
            os.path.join(cif_directory, fname), os.path.join(predict_dir, fname)
        )
        dummy_records.append({"id": mat_id, "dummy_target": 0.0})

    df = pd.DataFrame(dummy_records)
    df.to_csv(
        os.path.join(predict_dir, "id_prop.csv"), index=False, header=False
    )

    # Execute CGCNN predict.py script
    cmd = [
        sys.executable,
        os.path.join("cgcnn", "predict.py"),
        checkpoint_path,
        predict_dir,
    ]

    print("Running inference...")
    subprocess.run(cmd, check=True)

    # Read output results
    if os.path.exists("test_results.csv"):
        results_df = pd.read_csv(
            "test_results.csv", names=["material_id", "dummy", "predicted_value"]
        )
        results_df = results_df[["material_id", "predicted_value"]]
        print("\n--- Predictions ---")
        print(results_df.to_string(index=False))

        # Cleanup prediction working directory
        shutil.rmtree(predict_dir)


if __name__ == "__main__":
    # Point this to a directory containing the .cif files you want to predict
    predict_new_structures(cif_directory="cgcnn_dataset")