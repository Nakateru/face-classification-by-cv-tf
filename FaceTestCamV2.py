import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from resnet import resnet34
import face_recognition
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def jptext(img, moji, x, y):
    fontpath = "./DFLgs9.ttc"
    font = ImageFont.truetype(fontpath, 32)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y), moji, font=font, fill=(255, 255, 255))
    img = np.array(img_pil)
    return img


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


def put_text(image, name, prob, top, right, bottom, left):
    cv.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
    rectangle = cv.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv.FILLED)
    img = jptext(rectangle, name + ' ' + prob, left, bottom - 35)
    return img


if __name__ == '__main__':
    print('Face Classifier v2')
    print('Author  :  Nakateru (2020.09.03)')

    IMG_HEIGHT = 64
    IMG_WIDTH = 64
    CLASSES = ['name1', 'name2', 'name3']
    NUM_CLASSES = len(CLASSES)

    model = resnet34(NUM_CLASSES)
    model.build(input_shape=(None, IMG_HEIGHT, IMG_WIDTH, 3))
    model.load_weights('./log/model.ckpt')
    print('loaded weights')

    video_path = r"video_path"
    capture = cv.VideoCapture(video_path)
    while True:
        _, frame = capture.read()

        # cv.namedWindow('face_detect', cv.WINDOW_NORMAL)
        # cv.namedWindow('face_detect', cv.WINDOW_AUTOSIZE)
        # cv.resizeWindow('face_detect', 640, 360)

        face_locations = face_recognition.face_locations(frame)

        for top, right, bottom, left in face_locations:
            # print(top, right, bottom, left)

            frame_tf = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # BGR2RGB
            img_tf = tf.image.convert_image_dtype(frame_tf, tf.float32)
            img_crp = tf.image.crop_to_bounding_box(img_tf, top, left, bottom - top, right - left)
            img_tf_re = tf.image.resize(img_crp, [IMG_HEIGHT, IMG_WIDTH])  # resize image

            pred, prob = evaluate(img_tf_re)
            pred = pred.numpy()[0]
            # print('prediction:', pred)
            # print('probability:', prob)
            prob = prob.numpy()[0][pred]
            name = CLASSES[pred]

            # print('prediction:', name)
            # print('probability:', prob)

            frame = put_text(frame, name, str(prob), top, right, bottom, left)

        cv.imshow('face_detect', frame)
        cv.waitKey(10)
