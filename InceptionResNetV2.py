import os

from keras.layers import *
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.models import Sequential
from keras.optimizers import Adam

from Classifier import Classifier

class InceptionResNetV2(Classifier):
    def __init__(self):
        super().__init__()

    def create_model(self):
        base_model = InceptionResNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3),
            pooling='avg',
        )
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        base_model.trainable = False

        self.model = Sequential([
            base_model,
            Flatten(),
            Dense(self.CLASS_COUNT, activation='softmax'),
        ])

        opt = Adam(lr=0.0003)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

    def train(self, path_to_data, epoch_num = 10):
        super().train(self.model)
        hst = self.model.fit_generator(generator=self.train_data_gen,
                                  epochs=epoch_num,
                                  validation_steps=self.arr[1] // self.batch_size,
                                  steps_per_epoch=self.arr[0] // self.batch_size,
                                  validation_data=self.val_data_gen,
                                  shuffle=True)
        self.model.save('./models/InceptionResNetV2')
        return self.model

    def load_model(self, path):
        return super().load_model(path)

    def predict(self, img):
        return super().predict(img)
