# Импорт необходимых библиотек
import cv2  # OpenCV для работы с изображениями, детекторов и сопоставления
import numpy as np  # Библиотека для работы с массивами
import matplotlib.pyplot as plt  # Для построения гистограмм
import itertools  # Для перебора всех пар изображений

# Список путей к 5 изображениям (измените пути на те, которые соответствуют вашим файлам)
image_paths = ['images/1.jpg', 'images/2.jpg', 'images/3.jpeg', 'images/4.jpg', 'images/5.jpg']

# Загрузка изображений.
# Функция cv2.imread загружает изображения в BGR-формате.
images = [cv2.imread(path) for path in image_paths]

# Проверка, что все изображения успешно загружены
for idx, img in enumerate(images):
    if img is None:
        raise IOError(f"Ошибка при загрузке изображения: {image_paths[idx]}")

# Определим список значений параметра nfeatures.
# Этот параметр определяет максимальное число ключевых точек, которые ORB будет искать.
# Мы будем исследовать, как изменение этого параметра влияет на качество сопоставления.
nfeatures_list = [50, 100, 150, 200]

# Перебираем все уникальные пары изображений.
# Используем itertools.combinations, чтобы получить все пары (i, j) с i < j.
for (i, j) in itertools.combinations(range(len(images)), 2):
    # Выбор двух изображений для сопоставления
    img1 = images[i]
    img2 = images[j]

    # Преобразуем изображения в градации серого, так как большинство алгоритмов (например, ORB) работают с одноканальными изображениями.
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Создаём фигуру для построения гистограммы сопоставления для данной пары изображений.
    plt.figure(figsize=(10, 6))

    # Для каждого значения nfeatures будем получать ключевые точки, дескрипторы и выполнять сопоставление.
    for nfeatures in nfeatures_list:
        # Создание объекта детектора ORB с указанным числом ключевых точек.
        orb = cv2.ORB_create(nfeatures=nfeatures)

        # Обнаруживаем ключевые точки и вычисляем дескрипторы для обоих изображений.
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        # Создаём объект для сопоставления дескрипторов.
        # Так как ORB выдаёт бинарные дескрипторы, используем BFMatcher с Hamming-расстоянием.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Если дескрипторы не найдены для одного из изображений, пропускаем этот параметр.
        if des1 is None or des2 is None:
            print(
                f"Для nfeatures={nfeatures} не удалось найти дескрипторы в паре изображений {image_paths[i]} и {image_paths[j]}.")
            continue

        # Выполняем сопоставление дескрипторов между двумя изображениями.
        matches = bf.match(des1, des2)

        # Сортируем совпадения по расстоянию (меньшее расстояние – лучшее совпадение)
        matches = sorted(matches, key=lambda x: x.distance)

        # Извлекаем список расстояний из совпадений
        distances = [m.distance for m in matches]

        # Вычисляем минимальное расстояние из найденных совпадений
        min_distance = min(distances) if distances else None

        # Выводим в консоль информацию о минимальном расстоянии для текущей пары изображений и текущего значения nfeatures.
        print(
            f"Сравнение {image_paths[i]} и {image_paths[j]}, nfeatures={nfeatures}: минимальное расстояние = {min_distance}")

        # Строим гистограмму распределения расстояний для текущего параметра.
        # Для наглядности используем прозрачность (alpha=0.5), чтобы гистограммы разных настроек можно было сравнивать на одном графике.
        plt.hist(distances, bins=20, alpha=0.5, label=f'nfeatures={nfeatures}')

    # Добавляем заголовок и подписи осей к графику для данной пары изображений.
    plt.title(f"Гистограмма расстояний совпадений для пары: {image_paths[i]} и {image_paths[j]}")
    plt.xlabel("Расстояние")
    plt.ylabel("Частота")
    plt.legend()
    plt.show()
