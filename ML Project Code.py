ML Project Code 
pip install keras

# %%
from keras import backend as K
from keras.models import Model
from tensorflow.keras.layers import Conv1D
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# %%
import os 
print (os.getcwd())
os.chdir(r"c:\Users\sanba\OneDrive\Desktop\fall'24\ML\Project")

# %%


# %%
#load training data
train = json.load(open(r'train.json\data\processed\train.json'))

# %%
#load test data
test = json.load(open(r'test.json\data\processed\test.json'))

# %% [markdown]
# # Pre-Processs

# %% [markdown]
# Here we start building the input layer

# %%
def normalize(X1):
    '''normalizes to 0 to 1 scale'''
    norm = ((X1-X1.min())/(X1.max()-X1.min()))
    return norm

def preprocess(data):
    '''
    preprocesses the data and returns two list of processed data, 
    one normalized on 0 to 1 scale, dual channel picture, 85x85x2
    one normalized to power ratio, and combined, single channel picture, 85x85x1
    '''
    #make empty list for dataset normalized from 0 to 1
    X_data_scale = list()

    #make empty list for dataset normalized by power ratio
    X_data_power = list()

    #itterate over each element of the data
    for i in range(len(data)):
        #get elements of entry
        id = (data[i]['id'])
        B1 = np.reshape(data[i]['band_1'], (85,85))
        B2 = np.reshape(data[i]['band_2'], (85,85))
        
        #normalize to 0 to 1 range
        B1_norm = normalize(B1)
        B11_norm = normalize(B1)
        B2_norm = normalize(B2)
        norm_merge = np.dstack((B1_norm, B11_norm, B2_norm))
        
        #transform to power
        B1_power = np.power(10, np.divide(B1, 20))
        B2_power = np.power(10, np.divide(B2, 20))
        #power_merge = normalize(np.multiply(10,np.log10(np.add(np.power(10,np.divide(B1_power,10)),np.power(10,np.divide(B2_power,10))))))
        power_merge = np.dstack((B1_power,B2_power))

        #df.append({'id':id, 'image':norm_merge})
        #dp.append({'id':id, 'image':power_merge})
        X_data_scale.append(norm_merge)
        X_data_power.append(power_merge)
        
    return np.array(X_data_scale), np.array(X_data_power)

def get_classification(data):
    y_data = list()
    for i in range(len(data)):
        #id = (data[i]['id'])
        classification = (data[i]['is_iceberg'])
        #target.append({'id':id, 'is_iceberg':classification})
        y_data.append([classification])
    return(np.array(y_data))

def get_me_id(data):
    id = list()
    for i in range(len(data)):
        sample= (data[i]['id'])
        id.append([sample])
    return(np.array(id))

#def get_angle(data):
    angle = list()
    for i in range(len(data)):
        smp = (data[i]['inc_angle'])
        angle.append(smp)
    return(angle)

# %%
#preprocess training dataset
X_train_scale, X_train_power = preprocess(train)
y_train = get_classification(train)
train_id = get_me_id(train)

# %%
#preprocess testing dataset
X_test_scale, X_test_power = preprocess(test)
test_id = get_me_id(test)

# %% [markdown]
# # Split training & testing data

# %%
#split training & testing set
from sklearn.model_selection import train_test_split
X_train_d_scale, X_test_d_scale, y_train_d_scale, y_test_d_scale = train_test_split(X_train_scale, y_train, test_size=0.2, random_state=0)

# %%
#one hot encoding labels
#Y_train = to_categorical(y_train_d_scale, 2)
#Y_test = to_categorical(y_test_d_scale, 2)

# %% [markdown]
# # Model Building

# %% [markdown]
# ### Train model

# %%
#hyperparameters
batch_size = 16 #number of training examples to consider at once
num_epochs = 100 #itterate n times over training set
kernal_size = 3 #3x3 kernal
pool_size = 2 #2x2 pooling
conv_depth_1 = 32 # first layer 32 kernals
conv_depth_2 = 32 # first layer 32 kernals
conv_depth_3 = 64 # second layer 64 kernals
dense_1 = 64
dense_2 = 1
drop_out = 0.5 #drop out probability
hidden_size = 512 #number of neurons in fully-connected layer

# %%
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(conv_depth_1, (kernal_size, kernal_size), input_shape=(75,75,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Conv2D(conv_depth_2, (kernal_size, kernal_size)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Conv2D(conv_depth_3, (kernal_size, kernal_size)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(dense_1))
model.add(Activation('relu'))
model.add(Dropout(drop_out))
model.add(Dense(dense_2))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# %%
train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# %%
train_generator = train_datagen.flow(
        X_train_d_scale, y_train_d_scale,  # this is the target directory
        batch_size=batch_size)

# %%
history = model.fit(
    train_generator,
    steps_per_epoch=2000 // batch_size,
    epochs=num_epochs,
    validation_data=(X_test_d_scale, y_test_d_scale),
    validation_steps=800 // batch_size
)
model.save_weights('first_try.weights.h5')



# %% [markdown]
# ### Predict using model

# %%
dd=model.predict(X_test_scale,batch_size=batch_size,verbose=1)

# %%
dd

# %% [markdown]
# ### Output results

# %%
import pandas as pd

# Initialize an empty list to store each row
results_list = []

for i in range(len(test_id)):
    results_list.append({'id': test_id[i][0], 'is_iceberg': dd[i][0]})

# Convert the list of dictionaries to a DataFrame
results = pd.DataFrame(results_list)

# Save to CSV
results.to_csv("results.csv", index=False)


# %% [markdown]
# ### Plot history of accuracy and loss

# %%
import matplotlib.pyplot as plt

# Summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# %% [markdown]
# ### Plot model

# %%
model.summary()

# %%



