from datasets import load_dataset
import shutil
import os

# Define the directory where the dataset should be saved
lan = "eng"
download_dir = os.path.join("../env/data", lan)

# Create the directory if it does not exist, or remove it if it already exists
if os.path.exists(download_dir):
    shutil.rmtree(download_dir)
os.makedirs(download_dir)

# Load the dataset with the specified splits
dataset = load_dataset("SemRel/SemRel2024", lan, split=['train', 'dev', 'test'])

# Save each split to a CSV file in the specified directory
for split_name, split_data in zip(['train', 'dev', 'test'], dataset):
    split_data.to_csv(f"{download_dir}/{lan}_{split_name}.csv", index=False)

# Print a message indicating where the dataset has been saved and display the splits
print(f"Dataset downloaded and saved to {download_dir}")
for split_name in ['train', 'dev', 'test']:
    print(f"{lan}_{split_name} split saved to {download_dir}/{lan}_{split_name}.csv")
