import os
import subprocess
import sys

def setup_and_train():
    # 1. The script will see the "cgcnn" folder from your image and skip cloning
    if not os.path.exists("cgcnn"):
        print("Cloning CGCNN repository...")
        subprocess.run(
            ["git", "clone", "https://github.com/txie-93/cgcnn.git"], check=True
        )

    dataset_path = os.path.abspath("cgcnn_dataset_full")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset directory '{dataset_path}' missing."
        )

    # Ensures you have staged the target you want to train (bulk vs shear)
    if not os.path.exists(os.path.join(dataset_path, "id_prop.csv")):
        raise FileNotFoundError(
            "id_prop.csv is missing! Make sure to copy/rename either id_prop_bulk.csv or id_prop_shear.csv to 'id_prop.csv'."
        )

    # 2. Build CGCNN training command
    cmd = [
        sys.executable,
        os.path.join("cgcnn", "main.py"),
        dataset_path,
        "--task", "regression",
        "--epochs", "50",
        "--batch-size", "128",  # Increased to 128 for the large 13k dataset
        "--lr", "0.001",
        "--train-ratio", "0.8",
        "--val-ratio", "0.1",
        "--test-ratio", "0.1",
        "--workers", "0",       # CRITICAL FOR WINDOWS: Prevents PyTorch BrokenPipeError freezes
    ]

    print("Launching CGCNN Training...")
    print("Command:", " ".join(cmd))
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nTraining stopped or failed with error code {e.returncode}")

if __name__ == "__main__":
    setup_and_train()