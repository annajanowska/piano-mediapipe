import cvzone
import mediapipe as mp
import numpy as np
import cv2
import time

# Initialize width and height camera
cameraWidth = 1280
cameraHeight = 720

whiteNotes = ["4-c", "4-d", "4-e", "4-f", "4-g", "4-a", "4-h",
              "5-c", "5-c", "5-e", "5-f", "5-g", "5-a", "5-h",
              "6-c", "6-d", "6-e", "6-f", "6-g", "6-a", "6-h"]
blackNotes = ["4-cs", "4-ds", "4-fs", "4-gs", "4-as",
              "5-cs", "5-ds", "5-fs", "5-gs", "5-as",
              "6-cs", "6-ds", "6-fs", "6-gs", "6-as"]
buttonList = []


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
                if self.showHandLabel(handIndex, coordinatesLandmark, self.results):
                    label, coordinates = self.showHandLabel(handIndex, coordinatesLandmark, self.results)
                    cv2.putText(image, label, coordinates, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    def showHandLabel(self, index, coordinatesLandmark, results):
        label = results.multi_handedness[index].classification[0].label
        coordinates = tuple(np.multiply(
            np.array(
                (coordinatesLandmark.landmark[self.mpHands.HandLandmark.WRIST].x,
                 coordinatesLandmark.landmark[self.mpHands.HandLandmark.WRIST].y)),
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


class Button:

    def __init__(self, name, position, color, sound, size):
        self.name = name
        self.position = position
        self.color = color
        self.sound = sound
        self.size = size


def initializeKeyboard():
    noteHeight = 100
    noteWidth = 50
    for i in range(len(whiteNotes)):
        buttonList.append(Button(whiteNotes[i], [i * noteWidth + 100, noteHeight], [0, 0, 0], str(whiteNotes[i]) + ".mp3",
                                 [noteWidth, 3*noteHeight]))

    counter = 0
    tracer = 0
    checker = False

    for j in range(len(blackNotes)):
        tracer += 1
        buttonList.append(
            Button(blackNotes[j], [125 + (j * noteWidth) + (counter * noteWidth), noteHeight], [255, 255, 255],
                   str(blackNotes[j]) + ".mp3", [int(0.8*noteWidth), 2*noteHeight]))

        if tracer == 2 and checker is False:
            counter += 1
            tracer = 0
            checker = True

        if tracer == 3 and checker is True:
            counter += 1
            tracer = 0
            checker = False

    return buttonList


def showKeyboard(img, notesList):
    overlayImage = img.copy()

    for note in notesList:
        x, y = note.position
        width, height = note.size

        if note.color == [0, 0, 0]:
            cv2.rectangle(overlayImage, note.position, (x + width, y + height), (255, 255, 255), cv2.FILLED)
            cv2.rectangle(overlayImage, note.position, (x + width, y + height), (0, 0, 0), 2)
            cv2.putText(overlayImage, note.name, (x + 10, y + 250), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 4)

        if note.color == [255, 255, 255]:
            cv2.rectangle(overlayImage, note.position, (x + width, y + height), (0, 0, 0), cv2.FILLED)
            cv2.rectangle(overlayImage, note.position, (x + width, y + height), (0, 0, 0), 2)
            cv2.putText(overlayImage, note.name, (x + 10, y + 55), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 4)

    alpha = 0.4  # Factor of transparency

    img = cv2.addWeighted(overlayImage, alpha, img, 1-alpha, 0)

    return img


def showCamera():
    captureDevice = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    captureDevice.set(3, cameraWidth)
    captureDevice.set(4, cameraHeight)

    notesList = initializeKeyboard()

    previousTime = 0

    detector = HandDetector()

    while captureDevice.isOpened():
        success, img = captureDevice.read()
        img = detector.detectHands(img)
        img = showKeyboard(img, notesList)
        landmark = detector.findLandmarkList(img)

        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        cv2.putText(img, "Fps: " + str(int(fps)), (cameraWidth - 150, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.imshow("Hand Tracking", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if not captureDevice.isOpened():
        print("Camera is not connected properly!")
        exit()

    captureDevice.release()
    cv2.destroyAllWindows()


def initialize():
    showCamera()


if __name__ == "__main__":
    initialize()
