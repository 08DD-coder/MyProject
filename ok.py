import cv2
import mediapipe as mp
import numpy as np


def get_points(landmark, shape):
    points = []
    for mark in landmark:
        points.append([mark.x * shape[1], mark.y * shape[0]])
    return np.array(points, dtype=np.int32)


def palm_size(landmark, shape):
    x1, y1 = landmark[0].x * shape[1], landmark[0].y * shape[0]
    x2, y2 = landmark[5].x * shape[1], landmark[5].y * shape[0]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def hand_size(landmark, shape):
    x1, y1 = landmark[0].x * shape[1], landmark[0].y * shape[0]
    x2, y2 = landmark[12].x * shape[1], landmark[12].y * shape[0]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def _len(landmark, shape, a, b):
    x1, y1 = landmark[a].x * shape[1], landmark[a].y * shape[0]
    x2, y2 = landmark[b].x * shape[1], landmark[b].y * shape[0]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


handsDetector = mp.solutions.hands.Hands()


frame = cv2.imread("input.png")

flipped = np.fliplr(frame)
# переводим его в формат RGB для распознавания
flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
# Распознаем
results = handsDetector.process(flippedRGB)

all_points = get_points(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)
(x, y), r = cv2.minEnclosingCircle(get_points(results.multi_hand_landmarks[0].landmark, flippedRGB.shape))
dp = palm_size(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)
dh = hand_size(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)
thumb_len = _len(results.multi_hand_landmarks[0].landmark, flippedRGB.shape, 0, 4)
ind_finger_len = _len(results.multi_hand_landmarks[0].landmark, flippedRGB.shape, 0, 8)
mid_finger_len = _len(results.multi_hand_landmarks[0].landmark, flippedRGB.shape, 0, 12)
ring_finger_len = _len(results.multi_hand_landmarks[0].landmark, flippedRGB.shape, 0, 14)
pinky_len = _len(results.multi_hand_landmarks[0].landmark, flippedRGB.shape, 0, 18)
if 2 * r / dp < 1.35:
    print("Stone")
else:
    if ind_finger_len / mid_finger_len < 1.2 and (thumb_len / ring_finger_len < 1.2 and thumb_len / pinky_len < 1.2):
        print("Sissors")
    else:
        if 2 * r / dh < 1.2:
            print("Paper")

res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
cv2.imwrite("image.png", res_image)










#cap = cv2.VideoCapture(0)
#while(cap.isOpened()):
#    ret, frame = cap.read()
#    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
#        break
#    flipped = np.fliplr(frame)
#    # переводим его в формат RGB для распознавания
#    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
#    # Распознаем
#    results = handsDetector.process(flippedRGB)
#    # Рисуем распознанное, если распозналось
#    if results.multi_hand_landmarks is not None:
#        # нас интересует только подушечка указательного пальца (индекс 8)
#        # нужно умножить координаты а размеры картинки
#        x_tip = int(results.multi_hand_landmarks[0].landmark[8].x *
#                flippedRGB.shape[1])
#        y_tip = int(results.multi_hand_landmarks[0].landmark[8].y *
#                flippedRGB.shape[0])
#        cv2.circle(flippedRGB,(x_tip, y_tip), 10, (255, 0, 0), -1)
#        print(results.multi_hand_landmarks[0])
#    # переводим в BGR и показываем результат
#    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
 #   cv2.imshow("Hands", res_image)

# освобождаем ресурсы
#handsDetector.close()