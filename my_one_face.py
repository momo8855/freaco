import cv2 as cv
import numpy as np
import face_recognition as fc

def rescale_frame(frame, scale = .4):
    width = int(frame.shape[1] * scale)
    hight = int(frame.shape[0] * scale)
    dimensions = (width, hight)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


test = cv.imread('D:\images\me.jpg')
resize_test = rescale_frame(test)
cv.imshow('Me', resize_test)

face = fc.face_locations(resize_test)[0]
cv.rectangle(resize_test, (face[3], face[0]), (face[1], face[2]), (0, 255, 0), thickness=2)
cv.imshow('Me Detected', resize_test)

cv.waitKey(0)