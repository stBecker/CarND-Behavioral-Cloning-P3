import csv
import os
import cv2
import numpy as np
from keras import callbacks
from keras.layers import Lambda, Flatten, Dense, Cropping2D, Convolution2D, Dropout
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split


def read_images(images, angles, batch_sample):
    # create adjusted steering measurements for the side camera images
    center_angle = float(batch_sample[3])

    base_path = root_folder + '/dl_data/IMG/'
    cname = base_path + os.path.split(batch_sample[0].replace("\\", "/"))[-1]
    center_image = cv2.imread(cname)

    images.append(center_image)
    angles.append(center_angle)


root_folder = "/media/sb/exchange/CarND-Behavioral-Cloning-P3"
batch_size = 64
epochs = 5

samples = []
print("Reading driving_log")
with open(root_folder + '/dl_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
with open(root_folder + '/dl_data/driving_log2.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

#samples = samples[:1000]
print("preparing samples: ",len(samples))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print("Reading images")
images = []
angles = []
for batch_sample in train_samples:
    read_images(images, angles, batch_sample)
X_train = np.array(images)
y_train = np.array(angles)

images = []
angles = []
for batch_sample in validation_samples:
    read_images(images, angles, batch_sample)
X_val = np.array(images)
y_val = np.array(angles)

# Model adapted from the nVidia CNN (see writeup)
model = Sequential([
    # Preprocess incoming data, centered around zero with small standard deviation
    Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)),
    Lambda(lambda x: x / 255.0 - 0.5),
    Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"),
    Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"),
    Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"),
    Convolution2D(64, 3, 3, activation="relu"),
    Convolution2D(64, 3, 3, activation="relu"),
    Flatten(),
    Dense(100),
    Dense(50),
    Dense(10),
    Dense(1),
])
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'accuracy'])

checkpoint_cb = callbacks.ModelCheckpoint("weights/best_model.h5", save_best_only=True, period=1)
csv_logger = callbacks.CSVLogger('training.log')
# Use early stopping to prevent overfitting
early_stopping_cb = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

print("Training...")
history_object = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size,
                           nb_epoch=epochs, callbacks=[checkpoint_cb, csv_logger, early_stopping_cb])

print("saving model")
model.save('weights/model.h5')

# Uncomment to load a previously trained model
# model = load_model('weights/model.h5')
