import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(train_file_path, test_file_path, test_size=0.2, random_state=42):
    import numpy as np
    import json
    from sklearn.model_selection import train_test_split

    with open(train_file_path) as train_file:
        train_data = json.load(train_file)
    with open(test_file_path) as test_file:
        test_data = json.load(test_file)

    X_train = np.array([
        np.stack([np.array(instance['band_1'], dtype=np.float32).reshape(75, 75)], axis=-1)
        for instance in train_data
    ])
    y_train = np.array([instance.get('is_iceberg', 0) for instance in train_data], dtype=np.float32)

    X_test = np.array([
        np.stack([np.array(instance['band_1'], dtype=np.float32).reshape(75, 75)], axis=-1)
        for instance in test_data
    ])
    y_test = np.array([instance.get('is_iceberg', 0) for instance in test_data], dtype=np.float32)

    X_train_normalized = preprocess(X_train)
    X_test_normalized = preprocess(X_test)

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_normalized, y_train, test_size=test_size, random_state=random_state
    )

    return X_train_split, y_train_split, X_val_split, y_val_split, X_test_normalized, y_test


def preprocess(X):
    X_normalized = np.array([(X - np.min(X)) / (np.max(X) - np.min(X)) for X in X])
    return X_normalized

batch_size = 16
num_epochs = 50 # change to increase or decrease the number of epochs
kernel_size = 3 #
pool_size = 2
conv_depth_1 = 128
conv_depth_2 = 256
conv_depth_3 = 512
conv_depth_4 = 1024
dense_1 = 128
dense_2 = 1
drop_out = 0.2
weight_decay = 1e-4

def create_model():
    model = Sequential([
        Input(shape=(75, 75, 1)), 
        Conv2D(conv_depth_1, (kernel_size, kernel_size), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        Conv2D(conv_depth_2, (kernel_size, kernel_size), padding='same', kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        Conv2D(conv_depth_3, (kernel_size, kernel_size), padding='same', kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ## section can be uncommented to add another convolutional layer (3 --> 4) 
        Conv2D(conv_depth_4, (kernel_size, kernel_size), padding='same', kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        GlobalAveragePooling2D(), # uncomment to use, comment out Flatten()
        # Flatten(), 
        Dense(dense_1, kernel_regularizer=l2(weight_decay)),
        Activation('relu'),
        Dropout(drop_out),
        Dense(dense_2, activation='sigmoid')
    ])

    # compile model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_model(X_train, y_train, X_val, y_val, callbacks=None):
    if callbacks is None:
        callbacks = []

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
    model = create_model()

    # datagen = ImageDataGenerator(
    #     horizontal_flip=True,
    #     vertical_flip=True,
    #     rotation_range=15,    
    #     zoom_range=0.1,     
    #     width_shift_range=0.05, 
    #     height_shift_range=0.05,
    #     shear_range=10         
    # )
    # history = model.fit(
    #     datagen.flow(X_train, y_train, batch_size=batch_size),
    #     epochs=num_epochs,
    #     validation_data=(X_val, y_val),
    #     callbacks=[reduce_lr] + callbacks
    # )

    #comment out this section to use data augmentation
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(X_val, y_val),
        callbacks=[reduce_lr] + callbacks
    )

    model.save_weights('modified_model.weights.h5')

    return history

def evaluate_model(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    return test_loss, test_accuracy


def predict_image(model, preprocessed_image):
    prediction = model.predict(preprocessed_image)
    result = "Iceberg" if prediction[0] > 0.5 else "Not Iceberg"
    print("Prediction (Iceberg or Not Iceberg):", result)
    return result

