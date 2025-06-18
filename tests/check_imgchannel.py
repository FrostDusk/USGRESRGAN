from PIL import Image
import os

# Get first image file from the folder
folder_path = 'datasets/lrold'
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        try:
            img = Image.open(file_path)
            print(f"{file_name}: {img.mode}")
            break  # remove this if you want to check more images
        except PermissionError as e:
            print(f"Permission error: {e}")
