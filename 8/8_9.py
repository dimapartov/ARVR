import cv2        # OpenCV для работы с изображениями
import numpy as np  # NumPy для работы с массивами

# ------------------------- ШАГ 1: Загрузка изображений -------------------------
# Задаём пути к файлам изображений
big_image_path = 'images/1.jpg'  # путь к большому изображению
small_image_path = 'images/8_7.jpg'  # путь к меньшему изображению

# Загружаем изображения с помощью cv2.imread
big_image = cv2.imread(big_image_path)
small_image = cv2.imread(small_image_path)

# Проверяем, что оба изображения загружены корректно
if big_image is None:
    raise IOError(f"Не удалось загрузить большое изображение по пути: {big_image_path}")
if small_image is None:
    raise IOError(f"Не удалось загрузить маленькое изображение по пути: {small_image_path}")

# ------------------------- ШАГ 2: Определение положения наложения -------------------------
# Получаем размеры обоих изображений
(h_big, w_big) = big_image.shape[:2]        # размеры большого изображения (высота, ширина)
(h_small, w_small) = small_image.shape[:2]    # размеры маленького изображения (высота, ширина)

# Для наложения в центре вычисляем координаты верхнего левого угла маленького изображения
x_offset = (w_big - w_small) // 2  # координата x (отступ от левого края)
y_offset = (h_big - h_small) // 2  # координата y (отступ от верхнего края)

# ------------------------- ШАГ 3: Наложение изображения -------------------------
# Если требуется наложить изображение без учета прозрачности, можно выполнить простое копирование пикселей
# Копируем маленькое изображение в большую с вычисленными координатами
big_image[y_offset:y_offset+h_small, x_offset:x_offset+w_small] = small_image

# Если необходимо выполнить наложение с прозрачностью (blend), можно использовать функцию cv2.addWeighted.
# Например, для смешивания изображений с коэффициентами alpha и beta:
# alpha = 0.7   # вес большого изображения
# beta = 0.3    # вес маленького изображения
# region = big_image[y_offset:y_offset+h_small, x_offset:x_offset+w_small]
# blended = cv2.addWeighted(region, alpha, small_image, beta, 0)
# big_image[y_offset:y_offset+h_small, x_offset:x_offset+w_small] = blended

# ------------------------- ШАГ 4: Сохранение и отображение результата -------------------------
# Сохраняем полученное изображение в указанную директорию или с указанным именем
output_path = 'images/8_9.jpg'  # путь для сохранения результата

# Если директория не существует, создаём её с помощью модуля os
import os
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir) and output_dir != "":
    os.makedirs(output_dir)

# Сохраняем изображение
cv2.imwrite(output_path, big_image)
print(f"Результат наложения сохранён по пути: {output_path}")
