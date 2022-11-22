#%%------IMPORTS---------------------
"""
Created on Sat Sep 10 19:33:48 2022

@author: 
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#%%------LOAD DATASETS----------------------

BATCH_SIZE   =   32
IMAGE_SIZE   =   256
CHANNELS     =   1    # greyscale
FOLDER       =   './Data'

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    FOLDER,
    seed       =   123,
    shuffle    =   True,
    image_size =   (IMAGE_SIZE, IMAGE_SIZE),
    batch_size =   BATCH_SIZE,
    color_mode =  'grayscale'
)

class_names = dataset.class_names

# Seed shuffles the image datasets within gaussian distribution
# We are shuffling here to plot datasets in the next cell so that
# all cases can be seen

#%%------VISUZLIZE DATASET----------------------
for image_batch, labels_batch in dataset.take(1):
    print(image_batch.shape)
    print(labels_batch.numpy())
    
plt.figure(figsize=(10, 10))
for image_batch, labels_batch in dataset.take(1):
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8")[:,:,0], cmap='gray')
        plt.title(class_names[labels_batch[i]])
        plt.axis("off")
        
#%%------SPLIT, SHUFFLE AND RESCALE DATASETS----------------------

def get_dataset_partitions_tf(dataset, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10):
    assert (train_split + test_split + val_split) == 1
    
    ds_size = len(dataset)
    
    if shuffle:
        dataset = dataset.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    train_ds = dataset.take(train_size)    
    val_ds = dataset.skip(train_size).take(val_size)
    test_ds = dataset.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds

train, validation, test = get_dataset_partitions_tf(dataset)

resize_and_rescale = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
])

#%%------BUILDING CONVOLUTIONAL LAYERS----------------------

input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = len(class_names)

# Initialising the CNN
model = tf.keras.models.Sequential(resize_and_rescale)

# Step 1 - Convolution
model.add(tf.keras.layers.Conv2D(filters=32, padding="same", kernel_size=3, 
                                  activation='relu', input_shape = input_shape))

# Step 2 - Pooling
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
model.add(tf.keras.layers.Conv2D(filters=32, padding='same',kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
model.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection Neurons
model.add(tf.keras.layers.Dense(units=64, activation='relu'))

# Step 5 - Get one-hot matrix
model.add(tf.keras.layers.Dense(units = n_classes, activation = 'softmax'))

model.build(input_shape=input_shape)
model.summary()

#%%------COMPILING MODEL----------------------
model.compile(
            optimizer  =   'adam',
            loss       =   tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics    =   ['accuracy']
            )

EPOCHS  =  50
fitting = model.fit(train,
                    batch_size         =    BATCH_SIZE,
                    validation_data    =    validation,
                    verbose            =    1,
                    epochs             =    EPOCHS,
                    )

plt.figure(figsize=(8, 8), dpi=200)
plt.subplot(1, 2, 1)
plt.scatter(range(EPOCHS), fitting.history['accuracy'], label='Training Accuracy')
plt.scatter(range(EPOCHS), fitting.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.scatter(range(EPOCHS), fitting.history['loss'], label='Training Loss')
plt.scatter(range(EPOCHS), fitting.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#%%------PREDICT FROM MODEL----------------------

def predict(model, image, class_names):
    img_array = tf.keras.preprocessing.image.img_to_array(image.numpy())
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    # print(model.score(img_array))

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence
    
    
plt.figure(figsize=(15, 15))
for images, labels in test.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        predicted_class, confidence = predict(model        =  model, 
                                              image        =  images[i],
                                              class_names  =  class_names)
        actual_class = class_names[labels[i]] 
        
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
        
        plt.axis("off")

#%% Save model
model.save("models/CNN_%s.h5" %EPOCHS)

#%% Load Model
model = tf.keras.models.load_model('models/CNN_50.h5')