import cv2
import numpy as np

from camera.utils import calculate_naive_mask_bounding_box, get_largest_face_polygon
from machine import Cleaner, Controllers

cap = cv2.VideoCapture(0)

cascPath = "resources/haarcascade_frontalface_default.xml"
detector = faceCascade = cv2.CascadeClassifier(cascPath)

PLAYER_FACE_AREA_THRESHOLD = 0.1
# low_green = np.array([60, 52, 72])
low_green = np.array([60, 120, 120])
high_green = np.array([90, 255, 255])
low_red = np.array([161, 155, 84])
high_red = np.array([179, 255, 255])
low_blue = np.array([94, 80, 2])
high_blue = np.array([126, 255, 255])

cleaner = Cleaner()
controllers = Controllers()
controllers.enable()

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    last = {}

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    green_mask = cv2.inRange(hsv, low_green, high_green)
    green = cv2.bitwise_and(frame, frame, mask=green_mask)
    gx1, gy1, gx2, gy2 = calculate_naive_mask_bounding_box(green)
    largest_face_coordinates, largest_face_area = get_largest_face_polygon(faces)
    if largest_face_area < (PLAYER_FACE_AREA_THRESHOLD * frame.shape[0] * frame.shape[1]):
        cv2.putText(frame, "Come Closer", (largest_face_coordinates[0], largest_face_coordinates[1]),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255))
        cleaner.turn_off()
    else:
        if (gx1, gy1, gx2, gy2) != (0, 0, 0, 0):
            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 255, 255), 3)
            cleaner.turn_on()
        else:
            cleaner.turn_off()

        for face in faces:
            x1 = face[0]
            y1 = face[1]
            x2 = face[0] + face[2]
            y2 = face[1] + face[3]
            color = (0, 255, 0) if (x1, y1, x2, y2) == largest_face_coordinates else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
