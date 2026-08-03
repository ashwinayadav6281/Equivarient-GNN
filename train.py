import os
import shutil
import subprocess

# Ensure this matches your dataset folder name
DATA_DIR = "cgcnn_dataset_full" 

def run_training(target_name):
    print(f"\n{'='*50}")
    print(f"🚀 STARTING TRAINING: {target_name.upper()} MODULUS")
    print(f"{'='*50}")

    # 1. Copy the correct file to become 'id_prop.csv'
    source_csv = os.path.join(DATA_DIR, f"id_prop_{target_name}.csv")
    active_csv = os.path.join(DATA_DIR, "id_prop.csv")
    
    if not os.path.exists(source_csv):
        print(f"Error: Could not find {source_csv}")
        return

    shutil.copyfile(source_csv, active_csv)
    print(f"Prepared {active_csv} with {target_name} targets.")

    # 2. Execute your training script
    # Note: If your training script has a different name (e.g., main.py), change it here
    try:
        subprocess.run(["python", "train_cgcnn.py", DATA_DIR], check=True)
    except subprocess.CalledProcessError:
        print(f"Training failed for {target_name}. Stopping execution.")
        return

    # 3. Rename the output files so the next run doesn't overwrite them
    if os.path.exists("model_best.pth.tar"):
        os.replace("model_best.pth.tar", f"model_best_{target_name}.pth.tar")
        print(f"Saved best model as: model_best_{target_name}.pth.tar")
        
    if os.path.exists("checkpoint.pth.tar"):
        os.replace("checkpoint.pth.tar", f"checkpoint_{target_name}.pth.tar")

    print(f"✅ FINISHED TRAINING: {target_name.upper()} MODULUS\n")

if __name__ == "__main__":
    # 1. Train the Bulk Modulus model first
    run_training("bulk")
    
    # 2. Automatically train the Shear Modulus model right after
    run_training("shear")
    
    # 3. Clean up the temporary id_prop.csv file
    active_csv_path = os.path.join(DATA_DIR, "id_prop.csv")
    if os.path.exists(active_csv_path):
        os.remove(active_csv_path)
        
    print("🎉 ALL TRAINING COMPLETE!")