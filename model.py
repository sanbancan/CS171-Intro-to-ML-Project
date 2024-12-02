import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2
import matplotlib.pyplot as plt

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), input_shape=(75, 75, 1), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, kernel_regularizer=l2(1e-4)),
        Activation('relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    optimizer = Adam(learning_rate=0.001)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load and preprocess image
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
    img = cv2.resize(img, (75, 75)) 
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) 
    img = img.reshape(1, 75, 75, 1)
    return img

def load_trained_model(model, weights_path='modified_model.weights.h5'):
    model.load_weights(weights_path)
    print("Model weights loaded.")
    return model

def predict_image(model, image_path):
    preprocessed_image = load_and_preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    print("Prediction (Iceberg or Not Iceberg):", "Iceberg" if prediction[0] > 0.5 else "Not Iceberg")
    img = cv2.imread(image_path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Input Image")
    plt.show()

model = create_model()
model = load_trained_model(model)

# just for testing
if __name__ == "__main__":
    image_path = 'images/image_1.jpg'
    predict_image(model, image_path)
