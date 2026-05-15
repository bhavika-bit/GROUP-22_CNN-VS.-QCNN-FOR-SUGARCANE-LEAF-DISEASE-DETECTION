# PURE IMAGE PREPROCESSING PIPELINE

from google.colab import files
import zipfile
import os
import shutil
import cv2
import albumentations as A
from tqdm import tqdm

# Clean previous folders
input_dir = "/content/input_data"
output_dir = "/content/processed_data"

if os.path.exists(input_dir):
    shutil.rmtree(input_dir)
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Upload zip
uploaded = files.upload()
zip_name = list(uploaded.keys())[0]

# Extract zip
with zipfile.ZipFile(zip_name, 'r') as zip_ref:
    zip_ref.extractall(input_dir)

print("Extraction complete!")

# Define preprocessing transform
transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0, rotate_limit=0, scale_limit=0.2, p=1.0),
    A.CLAHE(p=0.3)
])

# Process images
for root, dirs, files_list in os.walk(input_dir):
    for file in tqdm(files_list):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):

            img_path = os.path.join(root, file)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            augmented = transform(image=image)
            processed_img = augmented["image"]

            # Convert back to BGR for saving
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)

            save_path = os.path.join(output_dir, file)
            cv2.imwrite(save_path, processed_img)

print("All images processed and saved!")

# Zip processed folder
shutil.make_archive("processed_images", 'zip', output_dir)

print("Processed zip ready for download!")
files.download("processed_images.zip")
