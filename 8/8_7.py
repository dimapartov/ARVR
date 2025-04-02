# Импорт необходимых библиотек:
import cv2           # OpenCV для обработки изображений и работы с алгоритмами детекции/сопоставления
import numpy as np   # NumPy для работы с массивами и математическими операциями
import matplotlib.pyplot as plt  # Matplotlib для построения графиков и гистограмм

# ------------------------- ШАГ 1: Загрузка и модификация изображения -------------------------

# Задаём путь к исходному изображению.
# Замените 'img1.jpg' на путь к вашему изображению.
img_path = 'images/1.jpg'
img_original = cv2.imread(img_path)

# Проверяем, что изображение загружено корректно.
if img_original is None:
    raise IOError(f"Не удалось загрузить изображение по пути: {img_path}")

# Задаём коэффициент масштабирования (например, уменьшаем изображение в 2 раза).
scale_factor = 0.5

# Создаём модифицированное изображение посредством уменьшения размера.
img_modified = cv2.resize(img_original, (0, 0), fx=scale_factor, fy=scale_factor)

save_dir = 'images/8_7.jpg'  # замените на нужную вам директорию
cv2.imwrite(save_dir, img_modified)


# Преобразуем изображения в оттенки серого (для детекции ключевых точек).
gray_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
gray_modified = cv2.cvtColor(img_modified, cv2.COLOR_BGR2GRAY)

# ------------------------- ШАГ 2: Обнаружение ключевых точек на исходном изображении -------------------------

# Создаём объект детектора ORB.
# Параметр nfeatures можно оставить по умолчанию или задать достаточно большое число, чтобы получить все ключевые точки.
orb = cv2.ORB_create(nfeatures=500)

# Обнаруживаем ключевые точки и вычисляем дескрипторы для исходного изображения.
kp_original, des_original = orb.detectAndCompute(gray_original, None)

# Если не удалось обнаружить ключевые точки, выводим сообщение.
if kp_original is None or len(kp_original) == 0:
    raise ValueError("Не удалось обнаружить ключевые точки на исходном изображении.")

# Вычисляем bounding box (прямоугольник) для ключевых точек исходного изображения.
# Для этого получаем координаты всех ключевых точек и находим минимальные и максимальные значения.
pts = np.array([kp.pt for kp in kp_original])
min_x = int(np.min(pts[:, 0]))
min_y = int(np.min(pts[:, 1]))
max_x = int(np.max(pts[:, 0]))
max_y = int(np.max(pts[:, 1]))

print(f"Исходный bounding box по ключевым точкам: ({min_x}, {min_y}) - ({max_x}, {max_y})")

# ------------------------- ШАГ 3: Обнаружение ключевых точек на модифицированном изображении с ограничением области -------------------------

# Обнаруживаем ключевые точки и вычисляем дескрипторы для модифицированного изображения.
kp_modified, des_modified = orb.detectAndCompute(gray_modified, None)

# Если не удалось обнаружить ключевые точки, выводим сообщение.
if kp_modified is None or len(kp_modified) == 0:
    raise ValueError("Не удалось обнаружить ключевые точки на модифицированном изображении.")

# Вычисляем масштабированный bounding box для модифицированного изображения.
# Так как изображение уменьшено, координаты bounding box нужно масштабировать.
min_x_mod = int(min_x * scale_factor)
min_y_mod = int(min_y * scale_factor)
max_x_mod = int(max_x * scale_factor)
max_y_mod = int(max_y * scale_factor)

print(f"Масштабированный bounding box для модифицированного изображения: ({min_x_mod}, {min_y_mod}) - ({max_x_mod}, {max_y_mod})")

# Фильтруем ключевые точки модифицированного изображения: оставляем только те,
# которые находятся внутри вычисленного масштабированного прямоугольника.
filtered_kp = []
filtered_des = []
for kp, desc in zip(kp_modified, des_modified):
    x, y = kp.pt
    if min_x_mod <= x <= max_x_mod and min_y_mod <= y <= max_y_mod:
        filtered_kp.append(kp)
        filtered_des.append(desc)

# Преобразуем список дескрипторов в массив NumPy.
if len(filtered_des) > 0:
    filtered_des = np.array(filtered_des)
else:
    raise ValueError("Нет ключевых точек в выбранном регионе модифицированного изображения.")

print(f"Количество ключевых точек на модифицированном изображении до фильтрации: {len(kp_modified)}")
print(f"Количество ключевых точек в пределах bounding box: {len(filtered_kp)}")

# ------------------------- ШАГ 4: Сопоставление дескрипторов между изображениями -------------------------

# Создаём объект BFMatcher.
# Поскольку ORB генерирует бинарные дескрипторы, используем норму Hamming.
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Выполняем сопоставление дескрипторов между исходным изображением и модифицированным (с ограниченными ключевыми точками).
matches = bf.match(des_original, filtered_des)

# Сортируем совпадения по расстоянию (меньшее расстояние соответствует лучшему совпадению).
matches = sorted(matches, key=lambda x: x.distance)

# Извлекаем список расстояний для каждого совпадения.
distances = [m.distance for m in matches]

# Вычисляем минимальное расстояние среди совпадений.
min_distance = min(distances) if distances else None
print(f"Минимальное расстояние между совпадающими дескрипторами: {min_distance}")

# ------------------------- ШАГ 5: Визуализация результатов -------------------------

# Строим гистограмму распределения расстояний совпадений дескрипторов.
plt.figure(figsize=(10, 6))
plt.hist(distances, bins=20, alpha=0.7, color='blue', edgecolor='black')
plt.title("Гистограмма расстояний дескрипторов между исходным и модифицированным изображениями")
plt.xlabel("Расстояние дескрипторов")
plt.ylabel("Частота")
plt.grid(True)
plt.show()
