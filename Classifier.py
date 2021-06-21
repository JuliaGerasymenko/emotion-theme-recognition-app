import numpy as np
import os

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
from tensorflow.keras.preprocessing import image

class Classifier:
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    def __init__(self):
        self.IMG_HEIGHT = 96
        self.IMG_WIDTH = 7409
        self. CLASS_COUNT = 56
        self.batch_size = 32
        self.path = os.path.join('/mnt/diploma/Emotion-and-Theme-Recognition-in-Music-Task/' 'data', 'data')

        self.train_dir = os.path.join(self.path, 'train')
        self.validation_dir = os.path.join(self.path, 'validation')
        self.subdirs = os.listdir(self.validation_dir)
        self.CLASS_COUNT = 56
        self.arr_images_num = []

        for total in [self.train_dir, self.validation_dir]:
            filename_num = 0
            for tag_folder in os.listdir(total):
                for img in os.listdir(os.path.join(total, tag_folder)):
                    filename_num += 1
            self.arr_images_num.append(filename_num)
        self.model = None

    def create_model(self):
        pass

    def train(self):
        self.model.trainable = False
        self.train_image_generator = ImageDataGenerator(rotation_range=0,
                                                        width_shift_range=0.6,
                                             horizontal_flip=True)

        self.validation_image_generator = ImageDataGenerator(rotation_range=0,
                                                             width_shift_range=0.6,
                                                             height_shift_range=0,
                                                             #                                      class_mode="categorical",
                                                             horizontal_flip=True)


        self.train_data_gen = self.train_image_generator.flow_from_directory(batch_size=self.batch_size,
                                                                             directory=self.train_dir,
                                                                             shuffle=True,
                                                                             target_size=(
                                                                             self.IMG_HEIGHT, self.IMG_WIDTH))

        self.val_data_gen = self.validation_image_generator.flow_from_directory(batch_size=self.batch_size,
                                                                                directory=self.validation_dir,
                                                                                shuffle=True,
                                                                                target_size=(
                                                                                self.IMG_HEIGHT, self.IMG_WIDTH))

        sample_training_images, _ = next(self.train_data_gen)

    def load_model(self, path):
        return load_model(path)

    def predict(self, img):
        img = img.resize((7409, 96))
        image_array = np.array(img)[:, :, 0:3]
        x = image.img_to_array(image_array)
        x = np.expand_dims(x, axis=0)
        return self.model.predict(x)
