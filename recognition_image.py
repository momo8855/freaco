import cv2 as cv
import numpy as np
import face_recognition as fc
from datetime import datetime

def rescale_frame(frame, scale = .4):
    width = int(frame.shape[1] * scale)
    hight = int(frame.shape[0] * scale)
    dimensions = (width, hight)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

images = [rescale_frame(cv.imread('D:\images\me.jpg')), cv.imread('D:\images\RTX6P9YW-1024x683.jpg'), cv.imread('D:\images\Bill.jpg'), cv.imread('D:\images\Abdel Rahman.jpg'),
                cv.imread('D:\images\M7md.jpg'), cv.imread('D:\images\Adel.jpg')]
names = ['Mostafa', 'Elon', 'Bill', 'Ghowil', 'Qtamesh', 'Dola']


def encoding(images):
    encodings = []
    for img in images:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encode = fc.face_encodings(img)[0]
        encodings.append(encode)

    return encodings

def mark_attendance(name):
    with open('attendance.csv', 'r+') as f:
        my_data_list = f.readline()
        name_list = []
        for line in my_data_list:
            entry = line.split(',')
            name_list.append(entry[0])
        if name not in name_list:
            now = datetime.now()
            dt_string = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dt_string}')


features = encoding(images)


img_to_detect = cv.imread('D:\images\elon-musk-left-bill-gates-right.jpg')
#img_to_detect = rescale_frame(cv.imread('D:\images\WhatsApp Image 2022-11-27 at 12.41.47 PM.jpeg'))
cv.imshow('To Detect', img_to_detect)
faces_to_detect = fc.face_locations(img_to_detect)
encodes_to_detect = fc.face_encodings(img_to_detect)

for encode_face, face in zip(encodes_to_detect, faces_to_detect):
        face_dis = fc.face_distance(features, encode_face)
        compare = fc.compare_faces(features, encode_face)
        print(face_dis)
        index = np.argmin(face_dis)
        print(index)
        name = names[index]
        print(name)
        if True in compare:
            cv.rectangle(img_to_detect, (face[3], face[0]), (face[1], face[2]), (0, 255, 0), thickness=2)
            cv.rectangle(img_to_detect, (face[3], face[0]-(face[0]-face[2])), (face[1], face[2]+35), (0, 255, 0), thickness=-1)
            cv.putText(img_to_detect, name, (face[3]+6,face[0]-(face[0]-face[2]-25)), cv.FONT_HERSHEY_COMPLEX, .8, (255,255,255), 2)
            mark_attendance(name)

cv.imshow('Detected', img_to_detect)
cv.waitKey(0)