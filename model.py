import numpy as np
import pandas as pd
import json
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Data loading and preprocessin
def load_data(train_file_path, test_file_path):
    """
    Load training and test data from JSON files and return as arrays.

    Parameters:
        train_file_path (str): Path to the training JSON file.
        test_file_path (str): Path to the test JSON file.

    Returns:
        tuple: X_train, y_train, X_test arrays
    """
    with open(train_file_path) as train_file:
        train_data = json.load(train_file)
    with open(test_file_path) as test_file:
        test_data = json.load(test_file)

    X_train = np.array([np.array(instance['band_1'], dtype=np.float32).reshape(75, 75)
                        for instance in train_data])
    y_train = np.array([instance['is_iceberg'] for instance in train_data], dtype=np.float32)
    X_test = np.array([np.array(instance['band_1'], dtype=np.float32).reshape(75, 75)
                       for instance in test_data])

    # Process test data
    X_test = np.array([
        np.stack(
            [np.array(instance['band_1'], dtype=np.float32).reshape(75, 75),
             np.array(instance['band_2'], dtype=np.float32).reshape(75, 75)],
            axis=-1
        ) for instance in test_data
    ])
    return X_train, y_train, X_test

def normalize(X):
    return (X - np.min(X)) / (np.max(X) - np.min(X))

def preprocess(X):
    X_normalized = np.array([normalize(image) for image in X])
    return X_normalized.reshape(-1, 75, 75, 1)

# Update the correct paths for your data files
train_file = r'dataset\train.json\data\processed\train.json'
test_file = r'dataset\test.json\data\processed\test.json'

X_train, y_train, X_test = load_data(train_file, test_file)
X_train = preprocess(X_train)
X_test = preprocess(X_test)

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Model hyperparameters
batch_size = 32
num_epochs = 50
kernel_size = 3
pool_size = 2
conv_depth_1 = 32
conv_depth_2 = 64
conv_depth_3 = 128
dense_1 = 128
dense_2 = 1
drop_out = 0.3
weight_decay = 1e-4

# Model architecture
model = Sequential([
    Conv2D(conv_depth_1, (kernel_size, kernel_size), input_shape=(75, 75, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)),
    # Conv2D(conv_depth_1, (kernel_size, kernel_size), input_shape=(75, 75, 1), padding='same'),
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
    GlobalAveragePooling2D(),
    # MaxPooling2D(pool_size=(pool_size, pool_size)),
    
    # Flatten(),
    # Dense(dense_1, kernel_regularizer=l2(weight_decay)),
    Dense(dense_1, kernel_regularizer=keras.regularizers.l2(weight_decay)),
    Activation('relu'),
    Dropout(drop_out),
    Dense(dense_2, activation='sigmoid')
])

# compile the model
optimizer = Adam(learning_rate=0.001)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

# learning rate scheduler for dynamically reducing learning rate 
def lr_scheduler(epoch, lr):
    return lr * 0.95 if epoch > 10 else lr
scheduler = LearningRateScheduler(lr_scheduler)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=num_epochs,
    validation_data=(X_val, y_val),
    callbacks=[reduce_lr]
)

model.save_weights('modified_model.weights.h5')

# accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Predict on test data (optional)
predictions = model.predict(X_test)
print(predictions[:5])

########################### second part ################################

# Train-test split for validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define model hyperparameters
batch_size = 32
num_epochs = 50
kernel_size = 3
pool_size = 2
conv_depth_1 = 32
conv_depth_2 = 64
conv_depth_3 = 128
dense_1 = 128
dense_2 = 1
drop_out = 0.3
weight_decay = 1e-4

# Model architecture with modifications
model = Sequential()
model.add(Conv2D(conv_depth_1, (kernel_size, kernel_size), input_shape=(75, 75, 1), padding='same', kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Conv2D(conv_depth_2, (kernel_size, kernel_size), padding='same', kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Conv2D(conv_depth_3, (kernel_size, kernel_size), padding='same', kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(GlobalAveragePooling2D())
# model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

# model.add(Flatten())
model.add(Dense(dense_1, kernel_regularizer=l2(weight_decay)))
model.add(Activation('relu'))
model.add(Dropout(drop_out))

model.add(Dense(dense_2, activation='sigmoid'))

# Compile the model
optimizer = Adam(learning_rate=0.001)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

# learning rate scheduler
def lr_scheduler(epoch, lr):
    return lr * 0.95 if epoch > 10 else lr
scheduler = keras.callbacks.LearningRateScheduler(lr_scheduler)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Model training
history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=num_epochs,
    validation_data=(X_val, y_val),
    # callbacks=[reduce_lr]
    callbacks=[reduce_lr, scheduler] # added scheduler
)

# Save model
model.save_weights('modified_model.weights.h5')

# accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Prediction example (optional)
predictions = model.predict(X_test)
print(predictions[:5])
