import cv2 as cv
import numpy as np
import face_recognition as fc


test = cv.imread('D:\images\860_main_beauty.png')
cv.imshow('all', test)
faces = fc.face_locations(test)
for face in faces:
    cv.rectangle(test, (face[3], face[0]), (face[1], face[2]), (0, 255, 0), thickness=2)

cv.imshow('All Detected', test)

cv.waitKey(0)