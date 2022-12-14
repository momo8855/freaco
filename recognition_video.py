import cv2 as cv
import numpy as np
import face_recognition as fc

def rescale_frame(frame, scale = .4):
    width = int(frame.shape[1] * scale)
    hight = int(frame.shape[0] * scale)
    dimensions = (width, hight)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

images = [rescale_frame(cv.imread('D:\images\me.jpg')), cv.imread('D:\images\RTX6P9YW-1024x683.jpg'), cv.imread('D:\images\Bill.jpg'),
                cv.imread('D:\images\Henry_Cavill_(48417913146)_(cropped).jpg'), cv.imread('D:\images\Millie-Bobby-Brown-ST-premiere.jpg')]
names = ['Mostafa', 'Bitch', 'Bill', 'Henry', 'Millie']


def encoding(images):
    encodings = []
    for img in images:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encode = fc.face_encodings(img)[0]
        encodings.append(encode)

    return encodings

features = encoding(images)

#cap = cv.VideoCapture(0)
cap = cv.VideoCapture('D:\images\holmes.mp4')

while True:
    success, img = cap.read()
    imgS = rescale_frame(img)
    encode_current = fc.face_encodings(imgS)
    faces = fc.face_locations(imgS)

    for encode_face, face in zip(encode_current, faces):
        face_dis = fc.face_distance(features, encode_face)
        compare = fc.compare_faces(features, encode_face)
        print(face_dis)
        index = np.argmin(face_dis)
        #print(index)
        name = names[index]
        print(name)
        if True in compare:
            cv.rectangle(imgS, (face[3], face[0]), (face[1], face[2]), (0, 255, 0), thickness=2)
            cv.rectangle(imgS, (face[3], face[0]-(face[0]-face[2])), (face[1], face[2]+35), (0, 255, 0), thickness=-1)
            cv.putText(imgS, name, (face[3]+6,face[0]-(face[0]-face[2]-25)), cv.FONT_HERSHEY_COMPLEX, .8, (255,255,255), 2)
        


    cv.imshow('Webcam',imgS)
    if cv.waitKey(20) & 0xff == ord('d'):
        break

cap.release()
cv.destroyAllWindows() 