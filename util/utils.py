import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

CHANNELS = (0, 1)
RANGES = (0, 180, 0, 256)
HIST_SIZE = (25, 25)


def calculate_face_histogram():
    cap = cv.VideoCapture(0)
    face_cascade = cv.CascadeClassifier(
        cv.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        _, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray)

        if len(faces) == 0:
            continue
        else:
            for (x, y, w, h) in faces:
                face_region = frame[y:y+h, x:x+w]
                # forehead_region = frame[y+10:y+20, x+20:x+w-20]
            cap.release()
            break

    hsv = cv.cvtColor(face_region, cv.COLOR_BGR2HSV)

    hist = cv.calcHist([hsv], CHANNELS, None, HIST_SIZE,
                       RANGES, accumulate=False)
    cv.normalize(hist, hist, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

    return face_region, hist


def calculate_skin_mask(hist):
    cap = cv.VideoCapture(0)

    while True:
        _, frame = cap.read()
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        skin_mask = cv.calcBackProject([hsv], CHANNELS, hist, RANGES, scale=1)
        # cv.threshold(skin_mask, 150, 255, cv.THRESH_BINARY)

        cv.imshow('skin mask', skin_mask)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
