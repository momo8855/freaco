import cv2 as cv
import numpy as np
import face_recognition as fc

test = cv.imread('D:\images\RTX6P9YW-1024x683.jpg')
cv.imshow('Img', test)
face = fc.face_locations(test)[0]
cv.rectangle(test, (face[3], face[0]), (face[1], face[2]), (0, 255, 0), thickness=2)
cv.imshow('Detected', test)

cv.waitKey(0)