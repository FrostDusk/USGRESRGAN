import os
import cv2
import numpy as np

from tqdm import tqdm

def degrade_image(img, scale=4):
    # Downscale
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_LINEAR)
    
    # Blur
    blurred = cv2.GaussianBlur(small, (3, 3), 0)
    
    # Speckle noise
    noise = np.random.randn(*blurred.shape) * 5
    noisy = blurred + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # Upscale back to original
    lr_img = cv2.resize(noisy, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return lr_img

# Paths
gt_dir = 'D:\Subject\Widyatama\Subject\Semester 8 - TA\project\SRP_env_py310\Real-ESRGAN\inputs\hr'
lr_dir = 'D:\Subject\Widyatama\Subject\Semester 8 - TA\project\SRP_env_py310\Real-ESRGAN\inputs\lrold'

os.makedirs(lr_dir, exist_ok=True)

# Supported extensions
valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')

print("Generating degraded LR images...")

for filename in tqdm(os.listdir(gt_dir)):
    if not filename.lower().endswith(valid_exts):
        continue

    gt_path = os.path.join(gt_dir, filename)
    img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"[Warning] Failed to read: {filename}")
        continue
    
    lr_img = degrade_image(img)
    
    out_path = os.path.join(lr_dir, filename)
    cv2.imwrite(out_path, lr_img)

print("✅ Done generating LR images.")