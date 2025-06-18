import os
import cv2
import numpy as np

from tqdm import tqdm

def degrade_image_v2(img, scale=4):
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_LINEAR)

    # Random Gaussian blur kernel size and sigma
    ksize = np.random.choice([3,5,7])
    sigma = np.random.uniform(0.1, 2.0)
    blurred = cv2.GaussianBlur(small, (ksize, ksize), sigma)

    # Add noise: either Gaussian or Poisson randomly
    if np.random.rand() > 0.5:
        noise = np.random.randn(*blurred.shape) * np.random.uniform(1,10)
        noisy = blurred + noise
    else:
        noisy = np.random.poisson(blurred).astype(np.float32)

    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    # JPEG compression artifacts simulation
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), np.random.randint(30, 95)]
    _, encoded_img = cv2.imencode('.jpg', noisy, encode_param)
    decoded = cv2.imdecode(encoded_img, cv2.IMREAD_GRAYSCALE)

    # Upscale back
    # lr_img = cv2.resize(decoded, (w, h), interpolation=cv2.INTER_LINEAR)

    return decoded

# Paths
gt_dir = 'D:\Subject\Widyatama\Subject\Semester 8 - TA\project\SRP_env_py310\Real-ESRGAN\inputs\hr'
lr_dir = 'D:\Subject\Widyatama\Subject\Semester 8 - TA\project\SRP_env_py310\Real-ESRGAN\inputs\lrnewx4'

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
    
    lr_img = degrade_image_v2(img)
    
    out_path = os.path.join(lr_dir, filename)
    cv2.imwrite(out_path, lr_img)

print("✅ Done generating LR images.")