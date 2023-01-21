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


handsDetector = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)


def what_on_hand(result, a):
    (x, y), r = cv2.minEnclosingCircle(get_points(result.multi_hand_landmarks[a].landmark, flippedRGB.shape))
    dp = palm_size(result.multi_hand_landmarks[a].landmark, flippedRGB.shape)
    dh = hand_size(result.multi_hand_landmarks[a].landmark, flippedRGB.shape)
    thumb_len = _len(result.multi_hand_landmarks[a].landmark, flippedRGB.shape, 0, 4)
    ind_finger_len = _len(result.multi_hand_landmarks[a].landmark, flippedRGB.shape, 0, 8)
    mid_finger_len = _len(result.multi_hand_landmarks[a].landmark, flippedRGB.shape, 0, 12)
    ring_finger_len = _len(result.multi_hand_landmarks[a].landmark, flippedRGB.shape, 0, 14)
    pinky_len = _len(result.multi_hand_landmarks[a].landmark, flippedRGB.shape, 0, 18)
    ring_finger_len1 = _len(result.multi_hand_landmarks[a].landmark, flippedRGB.shape, 0, 16)
    pinky_len1 = _len(result.multi_hand_landmarks[a].landmark, flippedRGB.shape, 0, 20)
    thumb_len1 = _len(result.multi_hand_landmarks[a].landmark, flippedRGB.shape, 0, 4)
    if x < 400:
        k = 1
    else:
        k = 2
    if 2 * r / dp < 1.5:
        return "Rock", k
    else:
        if ind_finger_len / mid_finger_len < 1.3 and (thumb_len / ring_finger_len < 1.3 and thumb_len / pinky_len < 1.3) \
                and not (pinky_len1 > pinky_len or ring_finger_len1 > ring_finger_len):
            return "Sissors", k
        else:
            if 2 * r / dh < 1.5:
                return "Paper", k


cap = cv2.VideoCapture(0)
wins1 = 0
wins2 = 0
i = 0
i1 = 0
i2 = 0
i3 = 0
i4 = 0
i5 = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break
    flipped = np.fliplr(frame)
    # переводим его в формат RGB для распознавания
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    # Распознаем
    results = handsDetector.process(flippedRGB)
    # Рисуем распознанное, если распозналось
    if results.multi_hand_landmarks is not None and len(results.multi_handedness) == 2:
        first, k1 = what_on_hand(results, 0)
        second, k2 = what_on_hand(results, 1)
        if k2 == 1:
            second, first = first, second
        elif k1 == 2:
            second, first = first, second
        print(first, second)

        if first == "Sissors" and second == "Paper":
            i += 1
            if i == 10:
                cv2.putText(flippedRGB, "First Player won!", (175, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 4)
                wins1 += 1
                while cv2.waitKey(1) & 0xFF != ord('e'):
                    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Hands", res_image)
                i = 0
        else:
            i = 0
        if first == "Paper" and second == "Sissors":
            i1 += 1
            if i1 == 10:
                cv2.putText(flippedRGB, "Second Player won!", (175, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 4)
                wins2 += 1
                while cv2.waitKey(1) & 0xFF != ord('e'):
                    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Hands", res_image)
                i1 = 0
        else:
            i1 = 0
        if first == "Rock" and second == "Paper":
            i2 += 1
            if i2 == 10:
                cv2.putText(flippedRGB, "Second Player won!", (175, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 4)
                wins2 += 1
                while cv2.waitKey(1) & 0xFF != ord('e'):
                    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Hands", res_image)
                i2 = 0
        else:
            i2 = 0
        if first == "Paper" and second == "Rock":
            i3 += 1
            if i3 == 10:
                cv2.putText(flippedRGB, "First Player won!", (175, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 4)
                wins1 += 1
                while cv2.waitKey(1) & 0xFF != ord('e'):
                    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Hands", res_image)
                i3 = 0
        else:
            i3 = 0
        if first == "Rock" and second == "Sissors":
            i4 += 1
            if i4 == 10:
                cv2.putText(flippedRGB, "First Player won!", (175, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 4)
                wins1 += 1
                while cv2.waitKey(1) & 0xFF != ord('e'):
                    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Hands", res_image)
                i4 = 0
        else:
            i4 = 0
        if first == "Sissors" and second == "Rock":
            i5 += 1
            if i5 == 10:
                cv2.putText(flippedRGB, "Second Player won!", (175, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 4)
                wins2 += 1
                while cv2.waitKey(1) & 0xFF != ord('e'):
                    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Hands", res_image)
                i5 = 0
        else:
            i5 = 0
    cv2.putText(flippedRGB, f"wins: {wins1}", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 4)
    cv2.putText(flippedRGB, f"wins: {wins2}", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 4)
    # переводим в BGR и показываем результат
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hands", res_image)
    if cv2.waitKey(1) & 0xFF == ord('w'):
        wins1 = 0
        wins2 = 0
# освобождаем ресурсы
handsDetector.close()