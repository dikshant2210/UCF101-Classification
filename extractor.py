from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
import numpy as np


class Extractor:
    def __init__(self):
        base_model = InceptionV3(
            weights='imagenet',
            include_top=True
        )
        self.model = Model(
            inputs=base_model.inputs,
            outputs=base_model.get_layer('avg_pool').output
        )

    def extract(self, image):
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        features = self.model.predict(image)
        features = features[0]
        return features
