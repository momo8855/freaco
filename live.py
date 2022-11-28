import cv2 as cv
import numpy as np
import face_recognition as fc

cap = cv.VideoCapture(0)

while True:
    success, img = cap.read()
    faces = fc.face_locations(img)

    for face in faces:
        cv.rectangle(img, (face[3], face[0]), (face[1], face[2]), (0, 255, 0), thickness=2)

    cv.imshow('Webcam',img)
    if cv.waitKey(20) & 0xff == ord('d'):
        break

cap.release()
cv.destroyAllWindows() 