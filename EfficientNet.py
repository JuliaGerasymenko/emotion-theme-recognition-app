import os
from PIL import ImageFile
from keras.layers import *
from keras.applications.efficientnet import EfficientNetB2
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from Classifier import Classifier

ImageFile.LOAD_TRUNCATED_IMAGES = True
class EfficientNet(Classifier):
    def __init__(self):
        super().__init__()

    def create_model(self):
        base_model = EfficientNetB2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3),
            pooling='avg',
        )
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        base_model.trainable = False

        self.model = Sequential([
            base_model,
            Dense(self.CLASS_COUNT, activation='softmax'),
        ])

        opt = Adam(lr=0.0001)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

    def train(self):
        super().train(self.model)
        hst = self.model.fit_generator(generator=self.train_data_gen,
                                  epochs=10,
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