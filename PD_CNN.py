#%%------IMPORTS---------------------
"""
Created on Sat Sep 10 19:33:48 2022
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

#%%------LOAD DATASETS----------------------

BATCH_SIZE   =   32
IMAGE_SIZE   =   256
CHANNELS     =   1    # greyscale
FOLDER       =   './Data/for_training'

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    FOLDER,
    seed       =   123,
    shuffle    =   True,
    image_size =   (IMAGE_SIZE, IMAGE_SIZE),
    batch_size =   BATCH_SIZE,
    color_mode =  'grayscale'
)

class_names =  dataset.class_names

dataset  =  dataset.take(10)

# Seed shuffles the image datasets within gaussian distribution
# We are shuffling here to plot datasets in the next cell so that all cases can be seen

#%%------VISUZLIZE DATASET----------------------
for image_batch, labels_batch in dataset.take(1):
    print(image_batch.shape)
    print(labels_batch.numpy())
    
plt.figure(figsize=(10, 10))
for image_batch, labels_batch in dataset.take(12):
    for i in range(4):
        ax = plt.subplot(2, 2, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8")[:,:,0], cmap='gray')
        plt.title(class_names[labels_batch[i]], fontsize=30)
        plt.axis("off")
        
#%%------SPLIT, SHUFFLE AND RESCALE DATASETS----------------------

def get_dataset_partitions_tf(dataset, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10):
    assert (train_split + test_split + val_split) == 1
    
    ds_size = len(dataset)
    
    if shuffle:
        dataset = dataset.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    print('train_size : %d' %train_size)
    print('test_size : %d' %val_size)
        
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
                                    strides=1,
                                    activation='relu', 
                                    input_shape = input_shape))

# Step 2 - Pooling
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# # Adding a second convolutional layer
# model.add(tf.keras.layers.Conv2D(filters=32, padding='same',kernel_size=3, 
#                                  activation='relu', strides=1))
# model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
model.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection Neurons
model.add(tf.keras.layers.Dense(units=32, activation='relu'))

# Step 5 - Get one-hot matrix
model.add(tf.keras.layers.Dense(units = n_classes, activation = 'softmax'))

model.build(input_shape=input_shape)
model.summary()

#%%------COMPILING MODEL----------------------
learning_rate  =  0.01
EPOCHS  =  50

model.compile(
            optimizer  =   tf.keras.optimizers.Adam(lr = learning_rate),
            loss       =   tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics    =   ['accuracy']
            )

start_time = time.time()
fitting = model.fit(train,
                    batch_size         =    BATCH_SIZE,
                    validation_data    =    validation,
                    verbose            =    1,
                    epochs             =    EPOCHS,
                    )
print("--- %s seconds ---" % (time.time() - start_time))

figsize = (8, 8)
fontsize  =  max(figsize)*1.8
plt.figure(figsize=(8, 8), dpi=200)
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), fitting.history['accuracy'][:EPOCHS], label='Training Accuracy')
plt.plot(range(EPOCHS), fitting.history['val_accuracy'][:EPOCHS], label='Validation Accuracy')
# plt.legend(loc='lower right', fontsize=fontsize)
plt.title('Accuracy', fontsize=fontsize)
plt.xlabel('Epochs', fontsize=fontsize)

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), fitting.history['loss'][:EPOCHS], label='Training')
plt.plot(range(EPOCHS), fitting.history['val_loss'][:EPOCHS], label='Validation')
plt.legend(loc='best', fontsize=fontsize)
plt.title('Loss', fontsize=fontsize)
plt.xlabel('Epochs', fontsize=fontsize)
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
    
    
figsize  =  [5,6]
fontsize =  min(figsize)*2
plt.figure(figsize = figsize, dpi=200)
for images, labels in test.take(1):
    for i in range(2):
        ax = plt.subplot(1, 2, i + 1)
        plt.imshow(images[i].numpy().astype("uint8")[:,:,0], cmap='gray')
        
        predicted_class, confidence = predict(model        =  model, 
                                              image        =  images[i],
                                              class_names  =  class_names)
        actual_class = class_names[labels[i]]         
        string  =  "Actual: %s\n Predicted: %s\n Confidence: %4.2f " \
                    %(actual_class, predicted_class, confidence) + '%'           
        # plt.text(x = 25, y = 240,
        #           s = string,
        #           fontsize=fontsize, color='white')
        plt.title(label = string, fontsize=fontsize, color='k')
        
        plt.axis("off")
        
plt.tight_layout()

#%% Save model
model.save("models/CNN_EPOCHS_%s.h5" %EPOCHS)

#%% Load Model
model = tf.keras.models.load_model('models/CNN_EPOCHS_50.h5')

#%% Plotting Loss and Accuracy

# EPOCHS = 50
figsize = (5, 5)
fontsize  =  min(figsize)*4
lw = min(figsize)*0.5
plt.figure(figsize=figsize, dpi=200)
# plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), fitting.history['accuracy'][:EPOCHS], 
         label='Training Accuracy', lw = lw)
plt.plot(range(EPOCHS), fitting.history['val_accuracy'][:EPOCHS], 
         label='Validation Accuracy', lw = lw)
# plt.legend(loc='lower right', fontsize=fontsize)
plt.ylabel('Accuracy', fontsize=fontsize*1.5)
# plt.xlabel('Epochs', fontsize=fontsize)
plt.xticks(ticks=np.linspace(0,50,3), fontsize=fontsize*1.5)
plt.yticks(ticks=np.linspace(0.7,1.2,6), fontsize=fontsize*1.5)
plt.xlim([0,50])
plt.ylim([0.75,1.05])

plt.tight_layout()
plt.show()

plt.figure(figsize=figsize, dpi=200)
# plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), fitting.history['loss'][:EPOCHS], 
         label='Training', lw = lw)
plt.plot(range(EPOCHS), fitting.history['val_loss'][:EPOCHS], 
         label='Validation', lw = lw)
plt.legend(loc='best', fontsize=fontsize)
plt.ylabel('Loss', fontsize=fontsize)
plt.xlabel('Epochs', fontsize=fontsize)
plt.xticks(ticks=np.linspace(0,50,6), fontsize=fontsize)
plt.yticks(ticks=np.linspace(0,1.6,5), fontsize=fontsize)
plt.xlim([0,50])
plt.ylim([0,1.6])

plt.tight_layout()
plt.show()

print('Model Score : %5.2f ' %(model.evaluate(test)[1]*100) + '%')
