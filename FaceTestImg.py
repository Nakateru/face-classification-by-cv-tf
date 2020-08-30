"""
FaceTest for image
tensorflow version >=2.2.0
CascadeClassifier reference https://github.com/opencv/opencv/tree/master/data/haarcascades
Custom parameters:
1.	faces names   classes=['name1', 'name2', 'name3']
	IMG_HEIGHT IMG_WIDTH
2. img_path	image path
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import cv2 as cv
from resnet import resnet34
import matplotlib.pyplot as plt
import numpy as np


def face_detect(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier("./haarcascades/haarcascade_frontalface_alt2.xml")
    faces = face_detector.detectMultiScale(gray, 1.11, 5, minSize=(60, 60))
    lenFaces = len(faces)
    # print(lenFaces)
    if lenFaces == 0:
        face_detector = cv.CascadeClassifier("./haarcascades/haarcascade_profileface.xml")
        faces = face_detector.detectMultiScale(gray, 1.09, 5, minSize=(60, 60))
    # print(faces)
    return faces


def evaluate(img):
    img = 2 * tf.cast(img, dtype=tf.float32) - 1
    img = tf.expand_dims(img, axis=0)
    # print(img) #(1, 64, 64, 3)
    logits = model(img)

    # print(logits)
    prob = tf.nn.softmax(logits, axis=1)  # probability
    # print(prob)
    pred = tf.argmax(prob, axis=1)  # prediction
    # print(pred)
    return pred, prob


def put_text(image, name, prob, y, x, w, h):
    img_retangle = cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv.putText(img_retangle, name + ' ' + str(prob), (int(x), int(y - 10)), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255),
               2)


if __name__ == '__main__':
    print('face classifier')
    print('Author  :  Nakateru (2020.08.31)')

    IMG_HEIGHT = 64
    IMG_WIDTH = 64
    CLASSES = ['name1', 'name2', 'name3']
    NUM_CLASSES = len(CLASSES)

    model = resnet34(NUM_CLASSES)
    model.build(input_shape=(None, IMG_HEIGHT, IMG_WIDTH, 3))
    model.load_weights('./log/model.ckpt')
    print('loaded weights')

    img_path = 'img_path'

    src = cv.imread(img_path)
    cv.namedWindow('face_detect', cv.WINDOW_AUTOSIZE)

    faces = face_detect(src)
    # print(faces)

    for i in faces:
        y, x, w, h = i[1], i[0], i[2], i[3]  # offset_height, offset_width, target_height, target_width
        # print(y, x, w, h)

        # cv => tf
        src_cvt = cv.cvtColor(src, cv.COLOR_BGR2RGB)  # BGR2RGB
        img_tf = tf.image.convert_image_dtype(src_cvt, tf.float32)  # convert to tf.tensor

        img_crp = tf.image.crop_to_bounding_box(img_tf, y, x, w, h)  # cropping
        img_tf_re = tf.image.resize(img_crp, [IMG_HEIGHT, IMG_WIDTH])  # resize image

        # print(img_tf.shape)  # (64, 64, 3)
        # imgplt = np.array(img_crp, np.int32)
        # plt.imshow(imgplt)
        # plt.show()

        pred, prob = evaluate(img_tf_re)
        pred = pred.numpy()[0]
        # print('prediction:', pred)
        # print('probability:', prob)
        prob = prob.numpy()[0][pred]
        name = CLASSES[pred]

        print('prediction:', name)
        print('probability:', prob)

        put_text(src, name, prob, y, x, w, h)

    cv.imshow('face_detect', src)

    cv.waitKey(0)
    cv.destroyAllWindows()
