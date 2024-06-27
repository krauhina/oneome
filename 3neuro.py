import tensorflow as tf
from tensorflow.keras import Sequential

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
TRAINING_DIR = "C:\\Users\\kigab\\PycharmProjects\\proj\\dataset\\X_train"
train_datagen = ImageDataGenerator(rescale=1.0 / 255.)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                              batch_size=40,
                              class_mode='binary',
                              target_size=(278,278))

VALIDATION_DIR = "C:\\Users\\kigab\\PycharmProjects\\proj\\dataset\\X_test"
validation_datagen = ImageDataGenerator(rescale=1.0 / 255.)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                      batch_size=40,
                                      class_mode='binary',
                                      target_size=(278,278))

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                           input_shape=(278,278, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(33, activation='softmax')
])
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(train_generator,
                           epochs=15,
                           verbose=1,
                           validation_data=validation_generator)

model.save('my_model.keras')