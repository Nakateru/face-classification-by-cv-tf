"""
FacesTrain

tensorflow version >=2.2.0

Custom parameters:
1.	NUM_CLASSES classes number
	BATCH_SIZE
	IMG_HEIGHT IMG_WIDTH
2. train_dir  training images directory
3. test_dir   testing  images directory
4. faces names   classes=['name1', 'name2', 'name3', 'name4', 'name5']

"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, Model, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from resnet import resnet18
import datetime


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    train_dir = ' train_dir'
    test_dir = ' test_dir'

    NUM_CLASSES = 5
    BATCH_SIZE = 40
    IMG_HEIGHT = 64
    IMG_WIDTH = 64

    train_image_generator = ImageDataGenerator(rescale=1. / 255,
                                               rotation_range=15,
                                               width_shift_range=0.05,
                                               height_shift_range=0.05,
                                               horizontal_flip=True,
                                               zoom_range=0.2,
                                               brightness_range=[0.3, 1.0]
                                               )
    train_data = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           classes=['name1', 'name2', 'name3', 'name4', 'name5'],
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           color_mode='rgb',
                                                           )

    test_image_generator = ImageDataGenerator(rescale=1. / 255,
                                              rotation_range=15,
                                              width_shift_range=0.05,
                                              height_shift_range=0.05,
                                              horizontal_flip=True,
                                              zoom_range=0.2,
                                              brightness_range=[0.3, 1.0]
                                              )

    test_data = test_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                         directory=test_dir,
                                                         classes=['name1', 'name2', 'name3', 'name4', 'name5'],
                                                         shuffle=True,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         color_mode='rgb',
                                                         )

    # print(train_data)
    # x, y = next(train_data)
    # plotImages(x[:5])
    # print(y)  # already one-hot

    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = './log/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    model = resnet18(NUM_CLASSES)
    model.build(input_shape=(None, IMG_HEIGHT, IMG_WIDTH, 3))
    model.summary()

    optimizer = optimizers.Adam(lr=1e-4)

    for step, (x, y) in enumerate(train_data):
        x = 2 * tf.cast(x, dtype=tf.float32) - 1
        y = tf.cast(y, dtype=tf.int32)
        with tf.GradientTape() as tape:
            # [b, IMG_HEIGHT, IMG_WIDTH, 3] => [b, n]
            logits = model(x)
            # print(logits.shape)  # (b, n)
            # print(y.shape)  # (b, n)

            # compute loss
            loss = tf.losses.categorical_crossentropy(y, logits, from_logits=True)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(step, 'loss:', float(loss))
            with summary_writer.as_default():
                tf.summary.scalar('train_loss', float(loss), step=step)

        if step % 300 == 0:
            total_num = 0
            total_correct = 0

            for step_test, (i, j) in enumerate(test_data):
                i = 2 * tf.cast(i, dtype=tf.float32) - 1
                j = tf.cast(j, dtype=tf.int64)
                j = tf.argmax(j, axis=1)
                logits = model(i)

                prob = tf.nn.softmax(logits, axis=1)
                pred = tf.argmax(prob, axis=1)
                # print(pred.dtype)
                # print(j.dtype)
                # print(pred.shape)
                # print(j.shape)

                correct = tf.cast(tf.equal(pred, j), dtype=tf.int32)
                correct = tf.reduce_sum(correct).numpy()

                total_num += i.shape[0]
                total_correct += correct

                if step_test == 100:
                    break

            acc = total_correct / total_num
            print('Test Accuracy:', acc)

            with summary_writer.as_default():
                 tf.summary.scalar('Test Accuracy', float(acc), step=step)

            model.save_weights('./log/model.ckpt')
            print('saved weights')
