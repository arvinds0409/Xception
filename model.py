import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
import numpy as np

# Define constants
image_size = (224, 224)  # Adjust image size for Xception
batch_size = 20
epochs = 2
num_classes = 3  # Number of classes

# Data augmentation and preprocessing for training
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,

)

# Load and prepare the training data
train_generator = train_datagen.flow_from_directory(
    'database/training',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Specify 'training' or 'validation'
)

# Load and prepare the validation data
validation_generator = train_datagen.flow_from_directory(
    'database/validation',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Specify 'training' or 'validation'
)

# Load the pre-trained Xception model
xception_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the pre-trained Xception model
for layer in xception_model.layers:
    layer.trainable = False

# Define custom layers for the classification head
flatten_layer = Flatten()(xception_model.output)
dense_layer = Dense(64)(flatten_layer)
normalize_layer = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(dense_layer)

# Output layer with 2 units for 2 classes and softmax activation
output_layer = Dense(num_classes, activation='softmax')(normalize_layer)

# Create the final model
face_recognition_model = Model(inputs=xception_model.input, outputs=output_layer)

# Compile the model
face_recognition_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = face_recognition_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs
)

# Save the trained model
face_recognition_model.save('face_recognition_model.h5')
