import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1) Setup
img_path = "images/8_3.png"
output_dir = "aruco"
os.makedirs(output_dir, exist_ok=True)

# 2) Load original
orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
h, w = orig.shape

# Variant 1: Original
cv2.imwrite(os.path.join(output_dir, "original.png"), orig)

# Variant 2: Gaussian blur
blurred = cv2.GaussianBlur(orig, (7, 7), 1.5)
cv2.imwrite(os.path.join(output_dir, "blurred.png"), blurred)

# Variant 3: Add Gaussian noise
noise = np.zeros_like(orig, dtype=np.float32)
cv2.randn(noise, 0, 30)  # mean=0, std=30
noisy = cv2.add(orig.astype(np.float32), noise)
noisy = np.clip(noisy, 0, 255).astype(np.uint8)
cv2.imwrite(os.path.join(output_dir, "gaussian_noise.png"), noisy)

# Variant 4: Histogram equalization
equalized = cv2.equalizeHist(orig)
cv2.imwrite(os.path.join(output_dir, "equalized.png"), equalized)

# Variant 5: Rotation (15 degrees)
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle=15, scale=1.0)
rotated = cv2.warpAffine(orig, M, (w, h), borderValue=255)
cv2.imwrite(os.path.join(output_dir, "rotated.png"), rotated)

# Dictionary of all variants
variants = {
    "Original": orig,
    "Blurred": blurred,
    "Noisy": noisy,
    "Equalized": equalized,
    "Rotated": rotated
}

# 3) Count contours vs threshold
thresholds = list(range(0, 256, 5))
counts = {name: [] for name in variants}

for name, img in variants.items():
    for th in thresholds:
        _, bin_img = cv2.threshold(img, th, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        counts[name].append(len(contours))

# 4) Plot
plt.figure(figsize=(10, 6))
for name in variants:
    plt.plot(thresholds, counts[name], label=name)
plt.xlabel("Threshold")
plt.ylabel("Number of contours")
plt.title("Contours detected vs Threshold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()