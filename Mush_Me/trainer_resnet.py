from google.cloud import storage
import pandas as pd
import numpy as np
import os, sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Recall
import joblib

import logging

from tensorflow.python.framework.ops import convert_to_tensor_v1
from tensorflow.python.keras.backend import dropout
from tensorflow.python.keras.layers.normalization import BatchNormalization
import google.cloud.logging

client = google.cloud.logging.Client()
client.get_default_handler()
client.setup_logging()

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'mush_me_hector2gt'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
# /!\ here you need to decide if you are going to train using the provided and uploaded data/train_1k.csv sample file
# or if you want to use the full dataset (you need need to upload it first of course)
BUCKET_TRAIN_DATA_PATH = './data/mini_train/'
BUCKET_TEST_DATA_PATH = './data/mini_test/'

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'mush_me_CNN'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

# not required here

### MODEL SPECS

batch_size = 64
img_size = (224,224)
epochs = 50
base_learning_rate = 0.005
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
    
    "Create Model Using "
    
    # Initialize the Pretrained Model
    feature_extractor = tf.keras.applications.ResNet50(weights='imagenet', 
                             input_shape=(224, 224, 3),
                             include_top=False)

    # Set this parameter to make sure it's not being trained
    feature_extractor.trainable = False

    # Set the input layer
    input_ = tf.keras.Input(shape=(224, 224, 3))
    
    # Data Augmentation
    
    x = data_augmentation(input_)

    # Set the feature extractor layer
    x = feature_extractor(input_, training=False)

    # Set the pooling layer
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Set the final layer with sigmoid activation function
    output_ = tf.keras.layers.Dense(179)(x)

    # Create the new model object
    model = tf.keras.Model(input_, output_)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_top_k_categorical_accuracy', 'accuracy'])
    
    es = keras.callbacks.EarlyStopping(patience = 7, restore_best_weights=False)
    
    model.fit(
            image_dataset_train,
            validation_data=image_dataset_val,
            epochs=epochs,
            callbacks = [es],
            verbose = 1
        )
    
    model.evaluate(image_dataset_test, batch_size = batch_size)
    
    return model


STORAGE_LOCATION = 'models/CNN/ResNet-50.h5'


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
    os.system(f'gsutil -m cp -r gs://{BUCKET_NAME}/data .')
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