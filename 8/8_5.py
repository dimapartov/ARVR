import os
import cv2
import cv2.aruco as aruco

input_dir = "aruco"
output_dir = os.path.join(input_dir, "aruco_marked")
os.makedirs(output_dir, exist_ok=True)

files = [
    "original.png",
    "blurred.png",
    "gaussian_noise.png",
    "equalized.png",
    "rotated.png"
]

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

for fname in files:
    path = os.path.join(input_dir, fname)
    image = cv2.imread(path)
    if image is None:
        print(f"❌ Файл не найден: {path}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None and len(ids) > 0:
        aruco.drawDetectedMarkers(image, corners, ids)

    if rejected is not None and len(rejected) > 0:
        aruco.drawDetectedMarkers(image, rejected, borderColor=(100, 0, 240))

    cv2.imwrite(os.path.join(output_dir, f"marked_{fname}"), image)

    # print(f"{fname}: Найдено маркеров: {0 if ids is None else len(ids)}")

print(f"\n✅ Обработка завершена. Отмеченные изображения сохранены в: {output_dir}")