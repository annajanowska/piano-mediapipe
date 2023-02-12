import mediapipe as mp
import numpy as np
import pandas as pd
from threading import Thread
from datetime import datetime
import cv2
import time
import pyglet
import glob
import os
import random
import timeit
import keyboard

# Initialize width and height camera
cameraWidth = 1280
cameraHeight = 720

# Define index of tips position
tipsId = [4, 8, 12, 16, 20]

# Initialize note names
whiteNotes = ["4-c", "4-d", "4-e", "4-f", "4-g", "4-a", "4-b",
              "5-c", "5-d", "5-e", "5-f", "5-g", "5-a", "5-b"]

blackNotes = ["4-cs", "4-ds", "4-fs", "4-gs", "4-as",
              "5-cs", "5-ds", "5-fs", "5-gs", "5-as"]


# Initialize settings for white key
widthWhiteNoteKey = 65
heightWhiteNoteKey = 330
shiftWhiteNote = 150
whiteColor = [0, 0, 0]

# Initialize settings for black key
widthBlackNoteKey = int(0.7 * widthWhiteNoteKey)
heightBlackNoteKey = int(2 / 3 * heightWhiteNoteKey)
shiftBlackNote = 150 + int(0.7 * widthWhiteNoteKey)
blackColor = [255, 255, 255]

buttonList = []
processDictionary = {}

class HandDetector:

    def __init__(self, mode=False, maxNumberHands=2, complexity=0, detectionConfidence=0.6, trackConfidence=0.5):
        self.mode = mode
        self.maxNumberHands = maxNumberHands
        self.complexity = complexity
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence
        self.results = None

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxNumberHands, self.complexity, self.detectionConfidence,
                                        self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def detectHands(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Mediapipe model works only with RGB mode
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        self.results = self.hands.process(image)  # Hand landmark detection process
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

    def showHandLabel(self, index, coordinates, result):
        label = result.multi_handedness[index].classification[0].label
        coordinates = tuple(np.multiply(
            np.array(
                (coordinates.landmark[self.mpHands.HandLandmark.WRIST].x,
                 coordinates.landmark[self.mpHands.HandLandmark.WRIST].y)),
            [cameraWidth, cameraHeight]).astype(int))

        return label, coordinates

    def findLandmarkList(self, img):
        landmarkList = []

        if self.results.multi_hand_landmarks:
            for handIndex, coordinates in enumerate(self.results.multi_hand_landmarks):
                label = self.results.multi_handedness[handIndex].classification[0].label
                myHand = self.results.multi_hand_landmarks[handIndex]
                handLandmark = []
                for index, lm in enumerate(myHand.landmark):
                    height, width, channels = img.shape
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    handLandmark.append([index, cx, cy, label])
                landmarkList.append(handLandmark)

        return landmarkList


class Button:

    def __init__(self, name, position, color, sound, size):
        self.name = name
        self.position = position
        self.color = color
        self.sound = sound
        self.size = size


class ThreadCountdown:

    def __init__(self):
        self.player = pyglet.media.Player()
        self.buttonSound = None

    def terminateProcess(self):
        self.player.pause()

    def runProcess(self, sound):
        path = sound
        MediaLoad = pyglet.media.load("../music/" + path)
        self.player.queue(MediaLoad)
        self.player.play()
        self.buttonSound = sound


def setCaptureDeviceSetting(cameraID=0):
    camera = cv2.VideoCapture(cameraID, cv2.CAP_DSHOW)
    camera.set(3, cameraWidth)
    camera.set(4, cameraHeight)

    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    return camera


def defineSoundTrackList():
    sounds = []
    os.chdir("../music")
    for file in glob.glob("*.mp3"):
        sounds.append(file)

    return sounds


def initializeKeyboard():
    soundsList = defineSoundTrackList()
    defineWhiteNoteKeys(soundsList)
    defineBlackNoteKeys(soundsList)

    return buttonList


def defineWhiteNoteKeys(musicSoundsList):
    for i in range(len(whiteNotes)):
        buttonList.append(
            Button(whiteNotes[i], [i * widthWhiteNoteKey + shiftWhiteNote, int(heightWhiteNoteKey / 3)], whiteColor,
                   defineSoundForSpecificKey(whiteNotes[i], musicSoundsList), [widthWhiteNoteKey, heightWhiteNoteKey]))


def defineBlackNoteKeys(musicSoundsList):
    counter = 0
    tracer = 0
    checker = False

    for i in range(len(blackNotes)):
        tracer += 1
        buttonList.append(
            Button(blackNotes[i],
                   [shiftBlackNote + (i * int(1.5 * widthBlackNoteKey)) + (counter * int(1.3 * widthBlackNoteKey)),
                    int(0.5 * heightBlackNoteKey)], blackColor,
                   defineSoundForSpecificKey(blackNotes[i], musicSoundsList),
                   [widthBlackNoteKey, heightBlackNoteKey]))

        if tracer == 2 and checker is False:
            counter += 1
            tracer = 0
            checker = True

        if tracer == 3 and checker is True:
            counter += 1
            tracer = 0
            checker = False


def defineSoundForSpecificKey(buttonName, soundsList):
    if buttonName + ".mp3" in soundsList:
        sound = buttonName + ".mp3"
    else:
        sound = "Not exist specific sound " + buttonName
        print(sound)

    return sound


def showKeyboard(img, notesList):
    overlayImage = img.copy()

    for note in notesList:
        x, y = note.position
        width, height = note.size

        if note.color == whiteColor:
            cv2.rectangle(overlayImage, note.position, (x + width, y + height), (255, 255, 255), cv2.FILLED)
            cv2.rectangle(overlayImage, note.position, (x + width, y + height), (0, 0, 0), 2)

        if note.color == blackColor:
            cv2.rectangle(overlayImage, note.position, (x + width, y + height), (0, 0, 0), cv2.FILLED)
            cv2.rectangle(overlayImage, note.position, (x + width, y + height), (0, 0, 0), 2)

    alpha = 0.5  # Factor of transparency

    img = cv2.addWeighted(overlayImage, alpha, img, 1 - alpha, 0)

    return img


def checkBendFingers(landmarkList, img):
    bendTipsList = []
    pressedButton = []

    if len(landmarkList) != 0:
        for i in range(len(landmarkList)):

            rightThumbIsBent = landmarkList[i][tipsId[0] - 1][1] - landmarkList[i][tipsId[0]][1] < 10 and \
                                landmarkList[i][tipsId[0]][3] == "Right";

            leftThumbIsBent = landmarkList[i][tipsId[0]][1] - landmarkList[i][tipsId[0] - 1][1] < 10 and \
                                landmarkList[i][tipsId[0]][3] == "Left";


            if rightThumbIsBent:
                bendTipsList.append(landmarkList[i][tipsId[0]])
                # print("Right thumb was bent")

            if leftThumbIsBent:
                bendTipsList.append(landmarkList[i][tipsId[0]])
                # print("Left thumb was bent")

            for index in range(1, 5):
                fingerIsBent = landmarkList[i][tipsId[index] - 2][2] - landmarkList[i][tipsId[index]][2] <= 35;
                if fingerIsBent:
                    bendTipsList.append(landmarkList[i][tipsId[index]])
                    # print("Finger was bent, id: " + str(id+1))

        pressedButton = checkIfButtonIsPressed(bendTipsList, img)

    return pressedButton, img


def checkIfButtonIsPressed(fingerBend, img):
    pressedButton = []

    if fingerBend:
        for button in buttonList:
            x, y = button.position
            width, height = button.size
            color = button.color

            for finger in fingerBend:
                pressedButtonColor = generateRandomColor()

                fingerOverKeyXCoord = x < finger[1] < x + width
                fingerOverWhiteKeyYCoord = y + heightBlackNoteKey < finger[2] < y + height
                fingerOverBlackKeyYCoord = y < finger[2] < y + height

                if color == whiteColor:
                    if fingerOverKeyXCoord and fingerOverWhiteKeyYCoord:
                        cv2.rectangle(img, button.position, (x + width, y + height), pressedButtonColor, cv2.FILLED)
                        pressedButton.append(button)
                else:
                    if fingerOverKeyXCoord and fingerOverBlackKeyYCoord:
                        cv2.rectangle(img, button.position, (x + width, y + height), pressedButtonColor, cv2.FILLED)
                        pressedButton.append(button)

    return pressedButton


def generateRandomColor():
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)
    color = (blue, green, red)

    return color


def createMusicFrameToPlay(pressedButtonList):
    currentMusicFrame = []

    for button in pressedButtonList:
        currentMusicFrame.append(button.sound)

    return currentMusicFrame


def initializeSystem():
    camera = setCaptureDeviceSetting()
    notes = initializeKeyboard()
    detector = HandDetector()

    return camera, notes, detector


def playBuildMusicForFrame(currentMusicFrameToPlay, previousMusicFrameToPlay):
    if currentMusicFrameToPlay == previousMusicFrameToPlay:
        pass
    elif currentMusicFrameToPlay != previousMusicFrameToPlay:

        soundToTurnOff = list(set(previousMusicFrameToPlay) - set(currentMusicFrameToPlay))

        if soundToTurnOff:
            for sound in soundToTurnOff:
                soundToKill = processDictionary.get(sound)
                Thread(target=soundToKill.terminateProcess).start()

        soundToPlay = list(set(currentMusicFrameToPlay) - set(previousMusicFrameToPlay))

        for note in soundToPlay:
            process = ThreadCountdown()
            t = Thread(target=process.runProcess, args=(note,))
            processDictionary[note] = process
            t.start()

def main():
    captureDevice, notesList, detector = initializeSystem()

    previousTime = 0
    previousMusicFrameToPlay = []

    while captureDevice.isOpened():
        success, img = captureDevice.read()
        img = detector.detectHands(img)
        img = showKeyboard(img, notesList)
        landmark = detector.findLandmarkList(img)
        pressedButtonList, img = checkBendFingers(landmark, img)

        currentMusicFrameToPlay = createMusicFrameToPlay(pressedButtonList)
        playBuildMusicForFrame(currentMusicFrameToPlay, previousMusicFrameToPlay)
        previousMusicFrameToPlay = currentMusicFrameToPlay

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

if __name__ == "__main__":
    main()
