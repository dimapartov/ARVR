import cv2
import numpy as np
import math

# Захватываем видео с веб-камеры
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Не удалось открыть видеокамеру")
    exit()

# Инициализируем детектор ORB и BFMatcher
orb = cv2.ORB_create(nfeatures=500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Переменные для референсного кадра (для сопоставления)
ref_frame = None
ref_kp = None
ref_des = None

print("Нажмите 'r', чтобы установить текущий кадр как референс для сопоставления.")
print("Нажмите 'q' для выхода.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр")
        break

    # Преобразуем кадр в оттенки серого
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаруживаем ключевые точки и дескрипторы в текущем кадре
    kp, des = orb.detectAndCompute(frame_gray, None)

    # Создаем копию кадра для вывода результатов
    display_frame = frame.copy()

    # Если референс установлен, выполняем сопоставление
    if ref_frame is not None and des is not None and ref_des is not None:
        # Сопоставляем дескрипторы текущего кадра с референсным
        matches = bf.match(ref_des, des)
        matches = sorted(matches, key=lambda x: x.distance)

        # Рассчитываем минимальное расстояние между совпадающими дескрипторами
        if matches:
            min_distance = min(m.distance for m in matches)
            cv2.putText(display_frame, f"Min Dist: {min_distance:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Анализируем параллельность линий: вычисляем углы между совпадающими ключевыми точками
        angles = []
        for m in matches:
            pt1 = ref_kp[m.queryIdx].pt  # ключевая точка из референсного кадра
            pt2 = kp[m.trainIdx].pt      # соответствующая точка из текущего кадра
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            angle = math.degrees(math.atan2(dy, dx))
            angles.append(angle)
        if angles:
            avg_angle = np.mean(angles)
            cv2.putText(display_frame, f"Avg Angle: {avg_angle:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Рисуем первые 20 совпадений для визуализации
        matched_img = cv2.drawMatches(ref_frame, ref_kp, frame, kp, matches[:20], None, flags=2)
        cv2.imshow("Matches", matched_img)

    # Пункт 9: Наложение уменьшенного изображения на текущее (отображается в правом верхнем углу)
    scale_factor = 0.3  # коэффициент уменьшения
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    h_small, w_small = small_frame.shape[:2]
    display_frame[0:h_small, display_frame.shape[1]-w_small:display_frame.shape[1]] = small_frame

    # Отображаем результирующий кадр
    cv2.imshow("Real-Time Processing", display_frame)

    # Обработка нажатия клавиш
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Устанавливаем текущий кадр как референс для сопоставления
        ref_frame = frame.copy()
        ref_gray = frame_gray.copy()
        ref_kp, ref_des = orb.detectAndCompute(ref_gray, None)
        print("Референсный кадр обновлен.")

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
