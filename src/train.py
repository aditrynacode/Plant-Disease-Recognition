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

test_ds = tf.keras.utils.image_dataset_from_directory(
    "data/test",
    image_size = img_size,
    batch_size = batch,
    shuffle = False
)

normalizer = tf.keras.layers.Rescaling(1/255)
training_ds = training_ds.map(lambda x, y: (normalizer(x), y))
validation_ds = validation_ds.map(lambda x, y: (normalizer(x), y))
test_ds = test_ds.map(lambda x, y: (normalizer(x), y))

AUTOTUNE = tf.data.AUTOTUNE

training_ds = training_ds.cache().prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = keras.Sequential([
    keras.layers.Conv2D(64, (3, 3), activation = 'relu', input_shape = (256, 256, 3)),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(3, activation = 'softmax')
])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(training_ds, validation_data = validation_ds, epochs = 10)

test_loss, test_acc = model.evaluate(test_ds)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)
