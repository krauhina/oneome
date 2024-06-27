emnist_labels = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 1025, 1105] + [i for i in range(1040, 1104)]
#emnist_labels = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]

import cv2
from tensorflow import keras
from tensorflow.keras import layers, models
from keras import optimizers
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape, LSTM, BatchNormalization
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from keras import backend as K
#from keras.constraints import maxnorm
import tensorflow as tf
from tensorflow.keras import Sequential


def emnist_model():
    model = Sequential()
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='valid', input_shape=(28, 28, 1), activation='relu'))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(emnist_labels), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model

import idx2numpy
import numpy as np

import gzip
import idx2numpy

emnist_path = 'archive\\emnist_source_files\\'

# Путь к сжатому файлу
X_train_compressed_file_path = emnist_path + 'emnist-byclass-train-images-idx3-ubyte.gz'
y_train_compressed_file_path = emnist_path + 'emnist-byclass-train-labels-idx1-ubyte.gz'
X_test_compressed_file_path = emnist_path + 'emnist-byclass-test-images-idx3-ubyte.gz'
y_test_compressed_file_path = emnist_path + 'emnist-byclass-test-labels-idx1-ubyte.gz'

# Путь, куда будет сохранен распакованный файл
X_train_decompressed_file_path = 'C:\\Users\\kigab\\PycharmProjects\\neuro\\emnist-byclass-train-images-idx3-ubyte.idx'
y_train_decompressed_file_path = 'C:\\Users\\kigab\\PycharmProjects\\neuro\\emnist-byclass-train-labels-idx1-ubyte.idx'
X_test_decompressed_file_path = 'C:\\Users\\kigab\\PycharmProjects\\neuro\\emnist-byclass-test-images-idx3-ubyte.idx'
y_test_decompressed_file_path = 'C:\\Users\\kigab\\PycharmProjects\\neuro\\emnist-byclass-test-labels-idx1-ubyte.idx'

# Распаковка файла
with gzip.open(X_train_compressed_file_path, 'rb') as f_in:
    with open(X_train_decompressed_file_path, 'wb') as f_out:
        f_out.write(f_in.read())

with gzip.open(y_train_compressed_file_path, 'rb') as f_in:
    with open(y_train_decompressed_file_path, 'wb') as f_out:
        f_out.write(f_in.read())

with gzip.open(X_test_compressed_file_path, 'rb') as f_in:
    with open(X_test_decompressed_file_path, 'wb') as f_out:
        f_out.write(f_in.read())

with gzip.open(y_test_compressed_file_path, 'rb') as f_in:
    with open(y_test_decompressed_file_path, 'wb') as f_out:
        f_out.write(f_in.read())
# Чтение распакованного файла с помощью idx2numpy
X_train = idx2numpy.convert_from_file(X_train_decompressed_file_path)
y_train = idx2numpy.convert_from_file(y_train_decompressed_file_path)
X_test = idx2numpy.convert_from_file(X_test_decompressed_file_path)
y_test = idx2numpy.convert_from_file(y_test_decompressed_file_path)


#X_train = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-train-images-idx3-ubyte')
#y_train = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-train-labels-idx1-ubyte.gz')
#X_test = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-test-images-idx3-ubyte.gz')
#y_test = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-test-labels-idx1-ubyte.gz')

X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1))
X_test = np.reshape(X_test, (X_test.shape[0], 28, 28, 1))

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, len(emnist_labels))

k = 1
X_train = X_train[:X_train.shape[0] // k]
y_train = y_train[:y_train.shape[0] // k]
X_test = X_test[:X_test.shape[0] // k]
y_test = y_test[:y_test.shape[0] // k]

# Normalize
X_train = X_train.astype(np.float32)
X_train /= 255.0
X_test = X_test.astype(np.float32)
X_test /= 255.0

x_train_cat = keras.utils.to_categorical(y_train, len(emnist_labels))
y_test_cat = keras.utils.to_categorical(y_test, len(emnist_labels))


learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
model = Sequential()
model = Sequential()
model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='valid', input_shape=(28, 28, 1), activation='relu'))
model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(emnist_labels), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# Обучение модели
model.fit(X_train, x_train_cat, validation_data=(X_test, y_test_cat), callbacks=[learning_rate_reduction], batch_size=64, epochs=30)
#model.fit(X_train, x_train_cat, validation_data=(X_test, y_test_cat)train, callbacks=[learning_rate_reduction], batch_size=64, epochs=30)

model.save('emnist_letters.h5')


model = keras.models.load_model('emnist_letters.h5')

def emnist_predict_img(model, img):
    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr/255.0
    img_arr[0] = np.rot90(img_arr[0], 3)
    img_arr[0] = np.fliplr(img_arr[0])
    img_arr = img_arr.reshape((1, 28, 28, 1))

    predict = model.predict([img_arr])
    result = np.argmax(predict, axis=1)
    return chr(emnist_labels[result[0]])

def letters_extract(image_file: str, out_size=28):
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    # Get contours
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    output = img.copy()

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
        # hierarchy[i][0]: the index of the next contour of the same level
        # hierarchy[i][1]: the index of the previous contour of the same level
        # hierarchy[i][2]: the index of the first child
        # hierarchy[i][3]: the index of the parent
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]
            # print(letter_crop.shape)

            # Resize letter canvas to square
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                # Enlarge image top-bottom
                # ------
                # ======
                # ------
                y_pos = size_max//2 - h//2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                # Enlarge image left-right
                # --||--
                x_pos = size_max//2 - w//2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            # Resize letter to 28x28 and add letter and its X-coordinate
            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    # Sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0], reverse=False)

    cv2.imshow("Input", img)
    # cv2.imshow("Gray", thresh)
    cv2.imshow("Enlarged", img_erode)
    cv2.imshow("Output", output)
    cv2.imshow("0", letters[0][2])
    cv2.imshow("1", letters[1][2])
    cv2.imshow("2", letters[2][2])
    cv2.imshow("3", letters[3][2])
    cv2.imshow("4", letters[4][2])
    cv2.waitKey(0)
    return letters


def img_to_str(model, image_file):
    letters = letters_extract(image_file)
    s_out = ""
    for i in range(len(letters)):
        dn = letters[i+1][0] - letters[i][0] - letters[i][1] if i < len(letters) - 1 else 0
        s_out += emnist_predict_img(model, letters[i][2])
        if (dn > letters[i][1]/4):
            s_out += ' '
    return s_out

model = keras.models.load_model('cycrillic_model.h5')
s_out = img_to_str(model, "../proj/FOTO4.jpg")
print(s_out)