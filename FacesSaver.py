"""
Faces saver
Set  3 directories:
# images dir
# cannot recognized dir
# saved faces dir

Summary
For saving faces for CNN training
If it can not be recognized,it will be rotated left or right 3 degs each try.

"""

import face_recognition
import cv2 as cv
import glob
import shutil


def rotate(image, angle, angle2):
    imgInfo = image.shape
    height = imgInfo[0]
    width = imgInfo[1]
    matRotate = cv.getRotationMatrix2D((height * 0.5, width * 0.5), angle, angle2)
    dst = cv.warpAffine(image, matRotate, (height, width))
    return dst


def save_face(face_image, img_path):
    # print(face_image.shape)
    img_name = img_path.split('\\')[-1]
    if not face_image.shape[0] < 60 and not face_image.shape[1] < 60:
        face_image = cv.cvtColor(face_image, cv.COLOR_BGR2RGB)
        cv.imwrite('./faces/saved/' + img_name, face_image)  # saved faces dir # ex: ./faces/saved/
    else:
        # print(img_name)
        shutil.move(img_path, './faces/NG/')  # cannot recognized dir # ex: ./faces/NG/


def face_fun(image):
    face_image = image
    face_locations = face_recognition.face_locations(image)
    # print(face_locations)
    lenFaces = len(face_locations)
    # print(lenFaces)
    for top, right, bottom, left in face_locations:
        face_image = image[top:bottom, left:right]

    return lenFaces, face_image


def detect_face(img_path):
    image = face_recognition.load_image_file(img_path)
    lenFaces, face_image0 = face_fun(image)

    if lenFaces == 0:
        # print('anticlockwise...')
        for i in range(3, 90, 3):  # anticlockwise
            rotatedImage = rotate(image, i, 0.6)
            lenFaces, face_image1 = face_fun(rotatedImage)
            if not lenFaces == 0:
                save_face(face_image1, img_path)
                break
            else:
                continue

        if lenFaces == 0:  # clockwise (if can not recognize when anticlockwise)
            # print('clockwise...')
            for i in range(-3, -90, -3):
                rotatedImage = rotate(image, i, 0.6)
                lenFaces, face_image2 = face_fun(rotatedImage)
                if not lenFaces == 0:
                    save_face(face_image2, img_path)
                    break
                else:
                    continue
        else:
            pass
        
        if lenFaces == 0:
            print('Can not recognize:')  # can not recognize both when clockwise and anticlockwise
            print(img_path)
            shutil.move(img_path, './faces/NG/')  # cannot recognized dir # ex: ./faces/NG/
        else:
            pass

    else:
        save_face(face_image0, img_path)

    #     cv_img = cv.cvtColor(face_image, cv.COLOR_RGB2BGR)
    #     cv.imshow('face', cv_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


if __name__ == '__main__':
    print('Face Saver')
    print('Author  :  Nakateru (2020.09.04)')

    file_list = glob.glob('./faces/*.jpg')  # images dir # ex: ./faces/*.jpg
    print(len(file_list), 'images')
    # print(file_list)

    for i in file_list:
        detect_face(i)

    print('Done')
