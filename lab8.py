import cv2
import numpy as np

image = cv2.imread('variant-9.png')
def pyramid(image, scale=1.5, minSize=(30, 30)):
    yield image
    while True:
        w = int(image.shape[1] / scale)
        image = cv2.resize(image, (w, w))

        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        yield image

for i, resized in enumerate(pyramid(image)):
    cv2.imshow(f'Layer {i}', resized)
    cv2.waitKey(0)

cv2.destroyAllWindows()


#Задание 2
cap = cv2.VideoCapture(0)
coordinates = []# Создаем пустой список для хранения координат метки

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)# Преобразуем изображение в оттенки серого и размываем его
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)# Применяем бинаризацию с порогом

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    # Находим контуры объектов на изображении

    # Отслеживаем большие контуры (метки)
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Вычисляем центр метки и добавляем его в список координат
            center_x = x + w // 2
            center_y = y + h // 2
            coordinates.append((center_x, center_y))

    # Отображение изображения с контурами
    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1)
    if key == 27:  # Нажатие клавиши ESC для выхода из программы
        break

# Вычисляем среднюю координату метки за текущий сеанс работы алгоритма
avg_x = sum([coord[0] for coord in coordinates]) / len(coordinates)
avg_y = sum([coord[1] for coord in coordinates]) / len(coordinates)
print("Средняя координата за сеанс работы: ({}, {})".format(avg_x, avg_y))

cap.release()
cv2.destroyAllWindows()