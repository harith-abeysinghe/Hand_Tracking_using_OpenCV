import cv2
import mediapipe as mp
import time

#Webcam
webcam = cv2.VideoCapture(0)


mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

previous_time = 0
current_time = 0


while 1:
    success, img = webcam.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:

            for id,landmark in enumerate(hand_landmark.landmark):
                # print(id, landmark)
                height, width, channel = img.shape
                cx,cy = int(landmark.x*width), int(landmark.y*height)
                print(id, cx, cy)


            mpDraw.draw_landmarks(img,hand_landmark, mpHands.HAND_CONNECTIONS)

    current_time = time.time()
    fps = 1/(current_time-previous_time)
    previous_time = current_time

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_TRIPLEX,3,(0,0,255),3)

    cv2.imshow("Camera", img)
    cv2.waitKey(1)
