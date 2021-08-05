import cv2
import mediapipe as mp
import time


# mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
# if len(x)==2:
#     d = ((x[1][0]-x[0][0])**2+(x[1][1]-x[0][1])**2)**0.5
#     print(d)
#
#     k = int(d/600 * 5)
#     if d > 300:
#         cv2.line(img,x[0],x[1], (0,0,255),k)
#     else:
#         cv2.line(img, x[0], x[1], (0, 0, 0), k)

class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=1):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        x = []
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo = 0, draw = 1):

        lmList = []

        if self.results.multi_hand_landmarks:
            # print(handNo)
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

                if draw :
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        return lmList

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0

    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        img = detector.find_hands(img)
        lmList = detector.findPosition(img,handNo=1)
        if lmList:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 120), cv2.FONT_ITALIC, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


main()
