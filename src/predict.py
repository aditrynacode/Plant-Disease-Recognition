import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("model/pdr_model.keras")

test_ds = tf.keras.utils.image_dataset_from_directory(
    "data/test",
    image_size = (224,224),
    batch_size = 32,
    shuffle = False
)

class_names = test_ds.class_names

normalizer = tf.keras.layers.Rescaling(1/255)
test_ds = test_ds.map(lambda x, y: (normalizer(x), y))

image = "data/test/powdery/81a63d7ef8245a72.jpg"

img = tf.keras.utils.load_img(image, target_size=(256, 256))
img = tf.keras.utils.img_to_array(img)   
img = img / 255          
img = tf.expand_dims(img, axis=0)      

prediction = model.predict(img)
predicted_number = np.argmax(prediction)
predicted_label = class_names[predicted_number]

plt.imshow(img[0])
plt.title(f"Predicted: {predicted_label}")
plt.axis("off")
plt.show() 