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

    image_name = 'image_name '
    img_path = './' + image_name + '.jpg'

    # Load an image
    image = face_recognition.load_image_file(img_path)
    face_locations = face_recognition.face_locations(image)

    for top, right, bottom, left in face_locations:
        # print(top, right, bottom, left)

        img_tf = tf.image.convert_image_dtype(image, tf.float32)
        img_crp = tf.image.crop_to_bounding_box(img_tf, top, left, bottom - top, right - left)
        img_tf_re = tf.image.resize(img_crp, [IMG_HEIGHT, IMG_WIDTH])  # resize image

        # print(pred, prob)
        # imgplt = np.array(img_crp, np.int32)
        # plt.imshow(img_crp)
        # plt.show()

        pred, prob = evaluate(img_tf_re)
        pred = pred.numpy()[0]
        # print('prediction:', pred)
        # print('probability:', prob)
        prob = prob.numpy()[0][pred]
        name = CLASSES[pred]

        print('prediction:', name)
        print('probability:', prob)

        image = put_text(image, name, str(prob), top, right, bottom, left)

        img = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        cv.namedWindow('face_detect', cv.WINDOW_NORMAL)
        cv.resizeWindow('face_detect', 1024, 769)
        cv.imshow('face_detect', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
