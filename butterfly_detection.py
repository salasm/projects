import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import os
import json
import re

# First of all, we will define basic settings for image size and training
img_height = 224
img_width = 224
batch_size = 32
epochs = 10
train_butterfly_dir = "C:/Users/mique/Downloads/train/butterfly"
train_no_butterfly_dir = "C:/Users/mique/Downloads/train/no butterfly"
test_dir = "C:/Users/mique/Downloads/test/test"
output_dir = "C:/Users/mique/nuwe-data-dl2/predictions"

# We set up image preprocessing and augmentation for training.
# It prepares the images by normalizing and adding random variations
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)
test_datagen = ImageDataGenerator(rescale=1./255)

'''
We create generators to load images from folders for training and validation,
which feed images to the model in batches. I divided the train file in two more files inside because
I thought manually labelling each image in butterfly or no butterfly would enable the model to be trained
and predict better. Also, I balanced the number of images with and without butterflies
to avoid over training the model with images of no butterflies,
I found out during the manual labelling there are more images without butterflies and predictions
were skewed towards predicting incorrectly more images with no butterfly.
'''
train_generator = train_datagen.flow_from_directory(
    "C:/Users/mique/Downloads/train",
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode ='binary',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    "C:/Users/mique/Downloads/train",
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode ='binary',
    subset = 'validation'
)

# We print the class indices to understand the label assignment
print("Class indices:", train_generator.class_indices)

# Here we will build a simple convolutional neural network (CNN), we will
# define the layers to process images and make predictions
model = Sequential([
    Conv2D(32, 3, padding = 'same', activation = 'relu', input_shape = (img_height, img_width, 3)),
    MaxPooling2D(),
    Conv2D(64, 3, padding = 'same', activation = 'relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding = 'same', activation = 'relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(64, activation = 'relu'),
    Dropout(0.5),
    Dense(1, activation = 'sigmoid')
])

# Now we compile the model with optimizer and loss function, which prepares the model for training
model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

# We train the model using the training data, this teaches the model to recognize butterflies
model.fit(
    train_generator,
    epochs = epochs,
    validation_data = validation_generator
)

# Now we prepare test images and make predictions to see if the image contains butterflies or not
test_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg'))]

# We create a function to sort files numerically, based on the number in the filename
def extract_number(filename):
    match = re.search(r'\d+', filename) # Extract the number from the filename ("image_123.jpg" --> 123)
    return int(match.group()) if match else float('inf')

# Sort files using the extracted number
test_files.sort(key = extract_number)  

predictions_dict = {"target": {}}
for filename in test_files:
    img_path = os.path.join(test_dir, filename)
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0
    prediction = model.predict(img_array)
    # Here initially, the logic was inverted, but it seemed like the model was predicting 0 when there is
    # a butterfly and 1 when there is not. So I decided to invert the following line of code to predict 1 for butterfly
    # and 0 for no butterfly.
    predicted_class = 0 if prediction[0][0] > 0.5 else 1
    predictions_dict["target"][filename] = predicted_class

# Finally, we save the predictions to the JSON file that will be evaluated
os.makedirs(output_dir, exist_ok = True)
output_path = os.path.join(output_dir, 'predictions.json')
with open(output_path, 'w') as f:
    json.dump(predictions_dict, f, indent = 4)
print(f"Predictions saved at: {output_path}")