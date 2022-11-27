#%%------IMPORTS---------------------
"""
Created on Sat Sep 10 19:33:48 2022
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#%%------LOAD DATASETS----------------------

BATCH_SIZE   =   100
IMAGE_SIZE   =   256
CHANNELS     =   1 # grayscale 
FOLDER       =   './Data/for_test/100'

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    FOLDER,
    shuffle    =   True,
    seed       =   123,
    image_size =   (IMAGE_SIZE, IMAGE_SIZE),
    batch_size =   BATCH_SIZE,
    color_mode =  'grayscale',
    label_mode =  'categorical'
)

class_names = dataset.class_names

#%%------VISUZLIZE DATASET----------------------
for image_batch, labels_batch in dataset.take(1):
    print(image_batch.shape)
    print(labels_batch.numpy().shape)
    
figsize  =  [10,10]
plt.figure(figsize = figsize, dpi=200)
for image_batch, labels_batch in dataset.take(1):
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)       
        plt.imshow(image_batch[i].numpy().astype("uint8")[:,:,0], cmap='gray')
        # plt.title(class_names[int(labels_batch[i].numpy()[0])])
        plt.title(class_names[int(np.argmax(labels_batch[i].numpy()))])
        plt.axis("off")
                
#%%------SPLIT, SHUFFLE AND RESCALE DATASETS----------------------

def get_dataset_partitions_tf(dataset, train_split=0.8):
        
    for image_batch, labels_batch in dataset:
        images   =  image_batch.numpy()/255  # Rescaling
        labels   =  labels_batch.numpy()
        dataset  =  [images, labels]
        
    ds_size = dataset[0].shape[0]
    
    train_size =  int(train_split * ds_size)
    train_ds   =  [dataset[0][:train_size,:], dataset[1][:train_size]]
    test_ds    =  [dataset[0][train_size:,:], dataset[1][train_size:]]
    plot_test  =  images[train_size:,:,:]
        
    return train_ds, test_ds, plot_test, dataset

train, test, plot_test, list_dataset = get_dataset_partitions_tf(dataset)

resize_and_rescale = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
])


#%%------BUILDING CONVOLUTIONAL LAYERS----------------------
input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = len(class_names)

# Initialising the CNN
model = tf.keras.models.Sequential(resize_and_rescale)

# Step 1 - Convolution
model.add(tf.keras.layers.Conv2D(filters=16, padding="same", kernel_size=3, strides=2,
                                  activation='relu', input_shape = input_shape))

# Step 2 - Pooling
# model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# # Adding a second convolutional layer
# model.add(tf.keras.layers.Conv2D(filters=32, padding='same',kernel_size=3, 
#                                  activation='relu', strides=2))
# model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
model.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection Neurons
model.add(tf.keras.layers.Dense(units=32, activation='relu'))

# Step 5 - Get one-hot matrix
model.add(tf.keras.layers.Dense(units = n_classes, activation = 'softmax'))

model.build(input_shape=input_shape)
model.summary()
model_feat = tf.keras.Model(inputs=model.input, outputs=model.get_layer(index=-1).output)


#%%  EXTRACT FEATURE FROM CNN TO FEED TO SVM
feat_train =  model_feat.predict(train[0])
feat_test  =  model_feat.predict(test[0])

#%%  FIT SVM
from sklearn.svm import SVC

svm = SVC(kernel='linear')
svm.fit(feat_train, np.argmax(train[1],axis=1))

a = svm.score(feat_test, np.argmax(test[1],axis=1))

b = svm.predict(feat_test)

#%%------PREDICT FROM MODEL----------------------

def predict(model, image, class_names, T=1):
    img_array = tf.keras.preprocessing.image.img_to_array(image)/255
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    # print(predictions[0])
    confidence = round(100 * (np.max(predictions[0])), 2)

    # As discussed with the TA

    # confidence  =  np.exp(predictions[0] * T) / np.sum(np.exp(predictions[0] * T))
    # print(confidence)
    # confidence  = round(max(confidence)*100, 2)
    # confidence = 1
    
    return predicted_class, confidence
    
figsize  =  [5,6]
fontsize =  min(figsize)*2
plt.figure(figsize = figsize, dpi=200)
for i in range(4):
    ax = plt.subplot(2, 2, i + 1)
    plt.imshow(test[0][i].astype("uint8")[:,:,0], cmap='gray')
    
    predicted_class, confidence = predict(model        =  model, 
                                          image        =  test[0][i],
                                          class_names  =  class_names)
    actual_class = class_names[np.argmax(test[1][i])]         
    string  =  "Actual: %s\n Predicted: %s\n Confidence: %4.2f " \
                %(actual_class, predicted_class, confidence) + '%'           
    # plt.text(x = 25, y = 240,
    #           s = string,
    #           fontsize=fontsize, color='white')
    plt.title(label = string, fontsize=fontsize, color='k')
    
    plt.axis("off")
        
plt.tight_layout()

print('Model Score : %5.2f ' %(model.evaluate(test)[1]*100) + '%')
        
#%% Save model
model.save("models/Support_vector_machine.h5")

#%% Load Model
model = tf.keras.models.load_model('models/Support_vector_machine_EPOCHS_50.h5')
