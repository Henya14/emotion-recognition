import keras
import tensorflow_hub as hub
import tensorflow as tf
from keras import layers
import os
import numpy as np
import pandas as pd
import datetime

class EmotionRecognitionModel(tf.keras.Model):
    def __init__(self, classes, *args, **kwargs):
        super(EmotionRecognitionModel, self).__init__(*args, **kwargs)
        self.feature_extractor = hub.KerasLayer("https://tfhub.dev/shoaib6174/swin_small_patch244_window877_kinetics400_1k/1")
        self.feature_extractor.trainable = False

        #self.Conv3d = layers.Conv3D(filters=16, kernel_size=3,activation="relu")
        #self.MaxPool3d = layers.MaxPool3D()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(units=16, activation="relu")
        self.dropout = layers.Dropout(0.4)
        self.dense2 = layers.Dense(8, activation="relu")
        self.dense3 = layers.Dense(len(classes), activation="softmax")
    
    
    def call(self, inputs):
        x = self.feature_extractor(inputs)
        #x = self.Conv3d(x)
        #x = self.MaxPool3d(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


def run_training(): 
    IMG_SIZE = 224
    MAX_SEQUENCE_LEN = 32
    NUM_OF_CHANNELS = 3
    LR = 1e-3
    EPOCH_NUM = 10

    train_df = pd.read_csv(os.path.join("data", "train.csv"))
    classes = np.unique(train_df["emotion"].values)
    transformer_model = EmotionRecognitionModel(classes)

    transformer_model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"])
    transformer_model.build(input_shape=(None, NUM_OF_CHANNELS, MAX_SEQUENCE_LEN, IMG_SIZE, IMG_SIZE))
    transformer_model.summary()

    t_df = tf.data.Dataset.load(os.path.join("data", "prepared_train_dataset")).map(lambda vid, label: (tf.cast((vid / 255.0), tf.float32), label))
    v_df = tf.data.Dataset.load(os.path.join("data", "prepared_validation_dataset")).map(lambda vid, label: (tf.cast((vid / 255.0), tf.float32), label))


    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    filepath = "/tmp/emotion_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, save_weights_only=True, save_best_only=True, verbose=1)

    EPOCH_NUM = 10
    transformer_model.fit(x=t_df.shuffle(200).batch(16), epochs=EPOCH_NUM, validation_data=v_df.batch(16), callbacks=[checkpoint, tensorboard_callback])

if __name__ == "__main__":
    run_training()