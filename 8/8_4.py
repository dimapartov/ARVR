import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def detect_aruco_markers(image):
    """Выделяет контуры ARUCO‑маркетов на изображении."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def count_aruco_markers_by_area(image, min_area, max_area=1e6):
    """Возвращает число контуров с площадью в диапазоне [min_area, max_area]."""
    contours = detect_aruco_markers(image)
    return sum(1 for cnt in contours if min_area <= cv2.contourArea(cnt) <= max_area)

# Параметры входа
input_dir = "aruco"
files = [
    "original.png",
    "blurred.png",
    "gaussian_noise.png",
    "equalized.png",
    "rotated.png"
]
titles = ["Original", "Blurred", "Noisy", "Equalized", "Rotated"]

# Диапазон площадей для анализа
area_thresholds = np.arange(0, 1000, 50)

# Собираем результаты
results = {title: [] for title in titles}
for fname, title in zip(files, titles):
    img = cv2.imread(os.path.join(input_dir, fname))
    for min_area in area_thresholds:
        results[title].append(count_aruco_markers_by_area(img, min_area))

# Построение графика
plt.figure(figsize=(12, 8))
for title in titles:
    plt.plot(area_thresholds, results[title], marker='o', label=title)

plt.xlabel("Минимальная площадь контура")
plt.ylabel("Количество найденных ARUCO-маркеров (контуров)")
plt.title("Зависимость числа маркеров от пороговой площади")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()