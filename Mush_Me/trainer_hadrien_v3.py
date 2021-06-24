from google.cloud import storage
import pandas as pd
import numpy as np
import os, sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import joblib

import logging
import google.cloud.logging

client = google.cloud.logging.Client()
client.get_default_handler()
client.setup_logging()

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'mush_me'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
# /!\ here you need to decide if you are going to train using the provided and uploaded data/train_1k.csv sample file
# or if you want to use the full dataset (you need need to upload it first of course)
BUCKET_TRAIN_DATA_PATH = './02_By_Species_Dataset/train/'
BUCKET_TEST_DATA_PATH = './02_By_Species_Dataset/test/'

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'mush_me_CNN'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v2'

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

# not required here

### MODEL SPECS

batch_size = 64
img_size = (224,224)
epochs = 50
base_learning_rate = 0.1
AUTOTUNE = tf.data.AUTOTUNE

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -



def get_data_train():
    """method to get train data from google cloud bucket"""
    image_dataset_train = tf.keras.preprocessing.image_dataset_from_directory(
        BUCKET_TRAIN_DATA_PATH, labels='inferred', label_mode='int',
        class_names=None, color_mode='rgb', batch_size=batch_size, image_size=img_size,
        shuffle=True, seed=17, validation_split=0.1, subset='training',
        interpolation='bilinear', follow_links=False
    )

    return image_dataset_train.prefetch(buffer_size=AUTOTUNE)

def get_data_val():
    """method to get validation data from google cloud bucket"""
    image_dataset_val = tf.keras.preprocessing.image_dataset_from_directory(
        BUCKET_TRAIN_DATA_PATH, labels='inferred', label_mode='int',
        class_names=None, color_mode='rgb', batch_size=batch_size, image_size=img_size,
        shuffle=True, seed=17, validation_split=0.1, subset='validation',
        interpolation='bilinear', follow_links=False
    )

    return image_dataset_val.prefetch(buffer_size=AUTOTUNE)

def get_data_test():
    """method to get test data from google cloud bucket"""


    image_dataset_test = tf.keras.preprocessing.image_dataset_from_directory(
        BUCKET_TEST_DATA_PATH, labels='inferred', label_mode='int',
        class_names=None, color_mode='rgb', batch_size=batch_size, image_size=img_size,
        shuffle=True, seed=None, validation_split=None, subset=None,
        interpolation='bilinear', follow_links=False
    )

    return image_dataset_test.prefetch(buffer_size=AUTOTUNE)


def train_model(image_dataset_train, image_dataset_val, image_dataset_test):
    """method that trains the model"""

    "Data Augmentation"

    data_augmentation = keras.Sequential(
        [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomContrast(0.1),
        layers.experimental.preprocessing.RandomRotation(0.2),
        layers.experimental.preprocessing.RandomZoom(0.2),
        ]
    )

    "Create Model Using MobileNet V2"

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    IMG_SHAPE = img_size + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')

    image_batch, label_batch = next(iter(image_dataset_train))
    feature_batch = base_model(image_batch)

    base_model.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)

    prediction_layer = tf.keras.layers.Dense(376)
    prediction_batch = prediction_layer(feature_batch_average)

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    es = keras.callbacks.EarlyStopping(patience = 4, restore_best_weights=True)

    model.fit(
            image_dataset_train,
            validation_data=image_dataset_val,
            epochs=epochs,
            callbacks = [es],
            verbose = 1
        )

    print("trained model")

    model.evaluate(image_dataset_test, batch_size = batch_size)

    return model


STORAGE_LOCATION = 'models/CNN/MobileNetV2_hadrien_v3.h5'


def upload_model_to_gcp():

    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(STORAGE_LOCATION)

    blob.upload_from_filename('model.h5')


def save_model(model):
    """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""

    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    model.save('model.h5')
    print("saved model locally")

    # Implement here
    upload_model_to_gcp()
    print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")


if __name__ == '__main__':
    # get data from GCP bucket
    logging.info('starting trainer.py')
    os.system(f'gsutil -m cp -r gs://{BUCKET_NAME}/02_By_Species_Dataset .')
    logging.info('copy file completed')

    logging.info(f'folders in dir {os.listdir()}')

    logging.info('preparing datasets')
    train_img = get_data_train()
    val_img = get_data_val()
    test_img = get_data_test()

    # train model (locally if this file was called through the run_locally command
    # or on GCP if it was called through the gcp_submit_training, in which case
    # this package is uploaded to GCP before being executed)
    MNV2 = train_model(train_img, val_img, test_img)

    # save trained model to GCP bucket (whether the training occured locally or on GCP)
    save_model(MNV2)
