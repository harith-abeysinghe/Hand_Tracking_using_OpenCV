import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, track_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode = self.mode, max_num_hands = self.max_hands, min_detection_confidence = self.detection_confidence,min_tracking_confidence=  self.track_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for hand_landmark in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_landmark, self.mpHands.HAND_CONNECTIONS)
        return img


    def find_postions(self,img, hand_num = 0, draw = True):
        landmark_list = []
        if self.results.multi_hand_landmarks:
            hand_landmark = self.results.multi_hand_landmarks[hand_num]
            for id, landmark in enumerate(hand_landmark.landmark):
                # print(id, landmark)
                height, width, channel = img.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                # print(id, cx, cy)
                landmark_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 15, (255,100,100), cv2.FILLED)
        return  landmark_list

def main():
    previous_time = 0
    current_time = 0

    # Webcam
    webcam = cv2.VideoCapture(0)

    detector = HandDetector()
    while 1:
        success, img = webcam.read()
        img = detector.find_hands(img)
        # landmark_list = detector.find_postions(img)

        # if len(landmark_list)!=0:
        #     print(landmark_list[5])

        key = cv2.waitKey(1)
        if key == 27:  # 27 is the ASCII code for 'Esc'
            break

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 255), 3)

        cv2.imshow("Camera", img)
        cv2.waitKey(1)
    # Release the webcam and close the OpenCV window
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

