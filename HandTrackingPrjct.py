import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2)

mpDraw = mp.solutions.drawing_utils

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    totalFingers = 0

    if results.multi_hand_landmarks:

        for handNo, handLms in enumerate(results.multi_hand_landmarks):

            lmList = []

            for id, lm in enumerate(handLms.landmark):

                h, w, c = img.shape

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                lmList.append([id, cx, cy])

            fingers = []

            # Detect Left or Right hand
            handLabel = results.multi_handedness[handNo].classification[0].label

            # Thumb
            if handLabel == "Right":

                if lmList[4][1] < lmList[3][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            else:  # Left hand

                if lmList[4][1] > lmList[3][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # Other four fingers
            for id in range(1, 5):

                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            totalFingers += fingers.count(1)

            mpDraw.draw_landmarks(
                img,
                handLms,
                mpHands.HAND_CONNECTIONS
            )

    cv2.putText(
        img,
        str(totalFingers),
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        3,
        (0, 0, 255),
        5
    )

    cv2.imshow("Image", img)
    cv2.waitKey(1)