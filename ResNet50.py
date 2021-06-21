import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.layers import *
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.models import Sequential
from Classifier import Classifier

class ResNet50(Classifier):
    def __init__(self):
        super().__init__()

    def create_model(self):
        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH ,3),
            pooling="avg",
        )
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        base_model.trainable = False
        self.model = Sequential([
            base_model,
            Flatten(),
            # GlobalAveragePooling1D(),
            Dropout(0.3),
            Dense(self.CLASS_COUNT, activation='softmax'),
        ])
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    def train(self):
        super().train(self.model)
        hst = self.model.fit_generator(generator=self.train_data_gen,
                                  epochs=6,
                                  validation_steps=self.arr[1] // self.batch_size,
                                  steps_per_epoch=self.arr[0] // self.batch_size,
                                  validation_data=self.val_data_gen,
                                  shuffle=True)
        self.model.save('./models/ResNet50')
        return self.model

    def load_model(self, path):
       return super().load_model(path)

    def predict(self, img):
        return super().predict(img)