import csv
import os
import time
import cv2
import numpy as np
import sklearn
from keras import callbacks
from keras.layers import Lambda, Flatten, Dense, Cropping2D, Convolution2D
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# root_folder = "c:"
root_folder = "/home/sb"
correction = 0.1  # this is a parameter to tune

batch_size = 32
epochs = 1

samples = []
print("Reading driving_log")
with open(root_folder + '/dl_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

samples = samples[:100]
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def read_images(images, angles, batch_sample):
    # create adjusted steering measurements for the side camera images
    center_angle = float(batch_sample[3])
    steering_left = center_angle + correction
    steering_right = center_angle - correction

    base_path = root_folder + '/dl_data/IMG/'
    cname = base_path + os.path.split(batch_sample[0].replace("\\", "/"))[-1]
    lname = base_path + os.path.split(batch_sample[1].replace("\\", "/"))[-1]
    rname = base_path + os.path.split(batch_sample[2].replace("\\", "/"))[-1]
    center_image = cv2.imread(cname)
    l_image = cv2.imread(lname)
    r_image = cv2.imread(rname)

    # image_flipped = np.fliplr(center_image)
    image_flipped = cv2.flip(center_image, 1)
    center_angle_flipped = -center_angle

    images.append(center_image)
    angles.append(center_angle)

    # images.extend([center_image, image_flipped, l_image, r_image])
    # angles.extend([center_angle, center_angle_flipped, steering_left, steering_right])


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                read_images(images, angles, batch_sample)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


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

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model = Sequential([
    # Preprocess incoming data, centered around zero with small standard deviation
    Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)),
    Lambda(lambda x: x / 255.0 - 0.5),
    # Flatten(),
    # Dense(1),
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
model.compile(loss='mse', optimizer='adam')

# training_run_timestamp = time.time()
checkpoint_cb = callbacks.ModelCheckpoint("weights/model.{epoch:02d}-{val_loss:.2f}.h5",
                                          monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False,
                                          mode='auto', period=1)
csv_logger = callbacks.CSVLogger('training.log')

print("Training...")
history_object = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size,
                           nb_epoch=epochs, callbacks=[checkpoint_cb, csv_logger])

# history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
#                                      validation_data=validation_generator,
#                                      nb_val_samples=len(validation_samples), nb_worker=4,
#                                      nb_epoch=3, callbacks=[checkpoint_cb, csv_logger])

print("saving model")
model.save('weights/model.h5')


# del model
# model = load_model('weights/model.h5')
y_pred = model.predict(X_val[:10], batch_size=batch_size)
print(y_pred)

# print("saving model")
# model.save('model_%s.h5' % training_run_timestamp)

### print the keys contained in the history object
print(history_object.history.keys())

# ### plot the training and validation loss for each epoch
# import matplotlib.pyplot as plt
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()
