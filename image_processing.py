import tensorflow as tf
from keras import layers
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
from constants import IMG_SIZE, MAX_SEQUENCE_LEN

IMAGE_CROP = layers.Cropping2D(cropping=(10, 70))

def crop_image(frame):
    cropped = IMAGE_CROP(frame[None, ...])
    cropped = cropped.numpy().squeeze()
    return cropped

def preprocess_video(path):
    print("Start", path)
    try: 
        cap = cv2.VideoCapture(path)
        frame_cnt = 0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Cut first ~1.3 seconds of the video
            if frame_cnt < 40:
                frame_cnt += 1
                continue
            
            frame = tf.image.resize(frame, (240,368))
            frame = crop_image(frame)
            frame = frame[:, :, [2, 1, 0]]
            frame = tf.image.resize(frame, (IMG_SIZE, IMG_SIZE)).numpy().astype(int)
            frames.append(frame)
    finally:
        cap.release()
    frames = np.array(frames)
    mask = np.zeros((MAX_SEQUENCE_LEN,))
    mask[:len(frames)] = 1
    if len(frames) > MAX_SEQUENCE_LEN:
        difference = len(frames) - MAX_SEQUENCE_LEN
        frames = frames[int(np.ceil(difference/3)):-int(np.floor(2*difference/3)), :, :,:]
    if len(frames) < MAX_SEQUENCE_LEN:
        frames = np.pad(frames,pad_width=((0,MAX_SEQUENCE_LEN-len(frames)), (0,0), (0,0), (0,0)), mode="constant")
    
    frames = np.transpose(frames, [3,0,1,2])
    return frames, mask

def create_label_processor(classes):
    return layers.StringLookup(num_oov_indices=0, vocabulary=classes)

