import cv2
import dlib
import numpy as np

from camera.utils import calculate_naive_mask_bounding_box, get_largest_face_polygon

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("resources/shape_predictor_68_face_landmarks.dat")

# low_green = np.array([60, 52, 72])
low_green = np.array([60, 120, 120])
high_green = np.array([90, 255, 255])
low_red = np.array([161, 155, 84])
high_red = np.array([179, 255, 255])
low_blue = np.array([94, 80, 2])
high_blue = np.array([126, 255, 255])

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    last = {}
    faces = detector(gray)
    green_mask = cv2.inRange(hsv, low_green, high_green)
    green = cv2.bitwise_and(frame, frame, mask=green_mask)
    gx1, gy1, gx2, gy2 = calculate_naive_mask_bounding_box(green)
    largest_face_coordinates = get_largest_face_polygon(faces)

    if (gx1, gy1, gx2, gy2) != (0, 0, 0, 0):
        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 255, 255), 3)
        # TODO - trigger cleaner

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        color = (0, 255, 0) if (x1, y1, x2, y2) == largest_face_coordinates else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # landmarks = predictor(gray, face)
        #
        # for n in range(68):
        #     x = landmarks.part(n).x
        #     y = landmarks.part(n).y
        #     last_x, last_y = last.get(n, (0, 0))
        #     # if n > 50:
        #         # print(f"{n} last: {(last_x, last_y)}; new: {(x, y)}")
        #     diff_x = abs(x-last_x)
        #     diff_y = abs(y-last_y)
        #     # print(f"{n} diff: {diff_x, diff_y}")
        #     color = (255, 0, 0) if diff_x > 2 or diff_y > 2 else (0, 0, 255)
        #     cv2.circle(frame, (x, y), 4, color, -1)
        #     if not (x == 0 and y == 0):
        #         # print(f"setting last[{n}]=({x},{y})")
        #         last[n] = (x, y)
    # keypoints = blober.detect(gray)
    # im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0, 0, 255),
    #                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow("Keypoints", im_with_keypoints)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
