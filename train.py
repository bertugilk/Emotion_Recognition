from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Flatten,Activation,Dropout,Conv2D,BatchNormalization
from keras_preprocessing.image import ImageDataGenerator

#CNN Model:
model=Sequential()

model.add(Conv2D(128,(5,5),padding = 'same',input_shape = (48,48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(5,5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(5,5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32,(5,5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(7))
model.add(Activation('softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen=ImageDataGenerator(1/255.0,validation_split=0.2)

# Training and testing data:
training_set=train_datagen.flow_from_directory("Dataset/train",
                                               target_size=(48,48),
                                               class_mode='categorical',
                                               batch_size=128,
                                               color_mode='grayscale',
                                               subset='training',
                                               shuffle=True)


test_set = test_datagen.flow_from_directory("Dataset/validation",
                                            target_size = (48,48),
                                            batch_size = 128,
                                            color_mode='grayscale',
                                            class_mode = 'categorical',
                                            subset='validation',
                                            shuffle=True)
# Training the model
model.fit_generator(generator=training_set,
                    steps_per_epoch=training_set.n // training_set.batch_size,
                    epochs=30,
                    validation_data=test_set,
                    validation_steps=test_set.n // test_set.batch_size)


testing_model = model.evaluate_generator(test_set, len(test_set), verbose=1)
print('Percentage of accuracy: ' + str(int(testing_model[1] * 10000) / 100) + '%')

model.save('Model/model.h5')