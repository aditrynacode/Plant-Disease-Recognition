import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

img_size = (256,256)
batch = 32
seed_value = 42

training_ds = tf.keras.utils.image_dataset_from_directory(
    "data/train",
    image_size = img_size,
    batch_size = batch,
    shuffle = True,
    seed = seed_value
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    "data/validation",
    image_size = img_size,
    batch_size = batch,
    shuffle = False
)

normalizer = tf.keras.Rescaling(1/255)
training_ds = training_ds.map(lambda x, y: (normalizer(x), y))
validation_ds = validation_ds.map(lambda x, y: (normalizer(x), y))

AUTOTUNE = tf.data.AUTOTUNE

training_ds = training_ds.cache().prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)
