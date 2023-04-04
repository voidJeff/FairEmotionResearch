'''
    Place this file in the same folder as the 'sessions' folder (or the equivalent thereof) and run.
    Each image with 'name.jpg' will be augmented into four 'name-augment_operation_name.jpg' and 
    placed into the same folder as the original
'''
import random
import numpy as np
import tensorflow as tf
from tensorflow import image
from tensorflow import io
import cv2
import glob

import os
import sys
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from PIL import Image

class Augment(object):
    def __init__(self):
        pass

    def augment(self, images, width, height, operations):
        augmented = []
        for i in range(len(images)):
            # aug = self.randomCrop(images[i], width, height)
            # aug = cv2.imread(input_file_path)
            aug = image.decode_png(io.read_file(input_file_path), channels=3)
            for j in range(4):
                
                print(img.dtype)
                if 'brightness' == operations[j]:
                    aug1 = self.random_brightness(image=aug,
                                                max_delta=0.2)
                    augmented.append(aug1)
                if 'contrast' == operations[j]:
                    aug2 = self.random_contrast(image=aug,
                                            lower=0.2,
                                            upper=1.8)
                    augmented.append(aug2)
                if 'hue' == operations[j]:
                    aug3 = self.random_hue(image=aug,
                                        max_delta=0.2)
                    augmented.append(aug3)
                if 'flip_h' == operations[j]:
                    aug4 = self.random_flip_h(image=aug)
                    augmented.append(aug4)

        # print('> Augmented images with {}'.format(operations), sep=' ')
        return augmented

    def randomCrop(self, img, width, height):
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        return img[y:y+height, x:x+width]

    def randomCropAll(self, imgs, width, height):
        imgs_l = []
        for i in range(imgs.shape[0]):
            image = imgs[i]
            image = self.randomCrop(image, width, height)
            # image = tf.minimum(image, 1.0)
            # image = tf.maximum(image, 0.0)
            imgs_l.append(image)
        return np.array(imgs_l)

    def random_hue(self, image, max_delta=0.5):
        return tf.image.random_hue(image=image,
                                   max_delta=max_delta)

    def random_contrast(self, image, lower, upper):
        return tf.image.random_contrast(image=image,
                                        lower=lower,
                                        upper=upper)

    def random_brightness(self, image, max_delta=0.5):
        return tf.image.random_brightness(image=image,
                                          max_delta=max_delta)

    def random_flip_h(self, image):
        return tf.image.random_flip_left_right(image=image)
    


for input_file_path in glob.iglob("sessions/**/*.*", recursive=True): # this assumes the folder is named sessions
    # Ignore non images
    # if not input_file_path.endswith((".jpg")):
    #     continue
    # input_file_path_example = 'sessions/6362/10059-angry_M-AS-01.jpg'
    img = cv2.imread(input_file_path)

    images = [img]  #TODO make simpler 
    # print('> Data Point Shape: {}'.format(img.shape))
    aug = Augment()
    operations = ['brightness', 'contrast', 'hue', 'flip_h']
    images_augmented = aug.augment(images, 144, 144, operations)
    # print("# output images: ", len(images_augmented))

    # output_dir = os.path.dirname(output_filepath)
    # Ensure the folder exists
    # os.makedirs(output_dir, exist_ok=True)

    for j,img in enumerate(images_augmented):
        tf_img = tf.io.encode_jpeg(
            img,
            format='',
            quality=100,
            progressive=False,
            optimize_size=False,
            chroma_downsampling=True,
            density_unit='in',
            x_density=300,
            y_density=300,
            xmp_metadata='',
            name=None
        )
        print(img.dtype, tf_img.dtype)
        output_file_name = input_file_path[:-4] + '-' + operations[j] + '.jpg'
        # plt.imshow(tf_img)
        # cv2.imshow('out', img)
        # cv2.imwrite(output_file_name, tf_img)
        # plt.savefig(output_file_name)
        # plt.imshow(img)
        # plt.show()
        tf.io.write_file(output_file_name, tf_img, name=None)

# cv2.waitKey(0)
# sys.exit()



