import mediapipe as mp
import numpy as np
import cv2
import time

# Initialize width and height camera
cameraWidth = 1280
cameraHeight = 720


class HandDetector:

    def __init__(self, mode=False, maxNumberHands=2, complexity=1, detectionConfidence=0.6, trackConfidence=0.5):
        self.mode = mode
        self.maxNumberHands = maxNumberHands
        self.complexity = complexity
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxNumberHands, self.complexity, self.detectionConfidence,
                                        self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def detectHands(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # model only works with RGB mode
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        self.results = self.hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.drawHandsConnections(image)

        return image

    def drawHandsConnections(self, image):
        if self.results.multi_hand_landmarks:
            for handIndex, coordinatesLandmark in enumerate(self.results.multi_hand_landmarks):
                self.mpDraw.draw_landmarks(image, coordinatesLandmark, self.mpHands.HAND_CONNECTIONS,
                                           self.mpDraw.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=4),
                                           self.mpDraw.DrawingSpec(color=(157, 168, 58), thickness=3, circle_radius=4),
                                           )
                if (self.showHandLabel(handIndex, coordinatesLandmark, self.results)):
                    label, coordinates = self.showHandLabel(handIndex, coordinatesLandmark, self.results)
                    cv2.putText(image, label, coordinates, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    def showHandLabel(self, index, coordinatesLandmark, results):
        label = results.multi_handedness[index].classification[0].label
        coordinates = tuple(np.multiply(
            np.array(
                (coordinatesLandmark.landmark[self.mpHands.HandLandmark.WRIST].x, coordinatesLandmark.landmark[self.mpHands.HandLandmark.WRIST].y)),
            [cameraWidth, cameraHeight]).astype(int))

        return label, coordinates

    def findLandmarkList(self, img):
        landmarkList = []
        if self.results.multi_hand_landmarks:
            for handIndex, coordinates in enumerate(self.results.multi_hand_landmarks):
                label = self.results.multi_handedness[handIndex].classification[0].label
                myHand = self.results.multi_hand_landmarks[handIndex]
                for id, lm in enumerate(myHand.landmark):
                     height, width, channels = img.shape
                     cx, cy = int(lm.x * width), int(lm.y * height)
                     landmarkList.append([id, cx, cy, label])

        return landmarkList


def showCamera():
    captureDevice = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    captureDevice.set(3, cameraWidth)
    captureDevice.set(4, cameraHeight)

    previousTime = 0

    detector = HandDetector()

    while captureDevice.isOpened():
        success, img = captureDevice.read()
        img = detector.detectHands(img)
        landmark = detector.findLandmarkList(img)

        currentTime = time.time()
        fps = 1/(currentTime-previousTime)
        previousTime = currentTime

        cv2.putText(img, "Fps: " + str(int(fps)), (cameraWidth-150, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.imshow("Hand Tracking", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if captureDevice.isOpened() != True:
        print("Camera is not connected properly!")
        exit()

    captureDevice.release()
    cv2.destroyAllWindows()


def initialize():
    showCamera()


if __name__ == "__main__":
    initialize()
