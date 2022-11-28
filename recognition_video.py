import cv2 as cv
import numpy as np
import face_recognition as fc

def rescale_frame(frame, scale = .4):
    width = int(frame.shape[1] * scale)
    hight = int(frame.shape[0] * scale)
    dimensions = (width, hight)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

images = [rescale_frame(cv.imread('D:\images\me.jpg')), cv.imread('D:\images\RTX6P9YW-1024x683.jpg'), cv.imread('D:\images\Bill.jpg')]
names = ['Mostafa', 'Bitch', 'Bill']


def encoding(images):
    encodings = []
    for img in images:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encode = fc.face_encodings(img)[0]
        encodings.append(encode)

    return encodings

features = encoding(images)

cap = cv.VideoCapture(0)

while True:
    success, img = cap.read()
    encode_current = fc.face_encodings(img)
    faces = fc.face_locations(img)

    for encode_face, face in zip(encode_current, faces):
        face_dis = fc.face_distance(features, encode_face)
        print(face_dis)
        index = np.argmin(face_dis)
        print(index)
        name = names[index]
        print(name)
        cv.rectangle(img, (face[3], face[0]), (face[1], face[2]), (0, 255, 0), thickness=2)
        cv.rectangle(img, (face[3], face[0]-(face[0]-face[2])), (face[1], face[2]+35), (0, 255, 0), thickness=-1)
        cv.putText(img, name, (face[3]+6,face[0]-(face[0]-face[2]-25)), cv.FONT_HERSHEY_COMPLEX, .8, (255,255,255), 2)
        


    cv.imshow('Webcam',img)
    if cv.waitKey(20) & 0xff == ord('d'):
        break

cap.release()
cv.destroyAllWindows() 