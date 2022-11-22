#%% ------IMPORTS---------------------
"""
Created on Tue Nov 22 09:44:02 2022

@author: Adi
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#%% ------LOAD DATASETS----------------------

BATCH_SIZE   =   200
IMAGE_SIZE   =   256
CHANNELS     =   1 # grayscale # 3    # RGB
FOLDER       =   './Data/for_test/200'

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    FOLDER,
    shuffle    =   True,
    seed       =   123,
    image_size =   (IMAGE_SIZE, IMAGE_SIZE),
    batch_size =   BATCH_SIZE,
    color_mode =  'grayscale'
)

class_names = dataset.class_names

# Seed shuffles the image datasets within gaussian distribution
# We are shuffling here to plot datasets in the next cell so that
# all cases can be seen

#%%------VISUZLIZE DATASET----------------------

for image_batch, labels_batch in dataset:
    images = image_batch.numpy()
    labels = labels_batch.numpy()
    print(image_batch.shape)
    print(labels_batch.numpy())
    
figsize  =  [10,10]
plt.figure(figsize = figsize, dpi=200)
for i in range(len(dataset)):
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)       
        plt.imshow(images[i,:,:,0], cmap='gray')
        plt.title(class_names[labels[i]])
        plt.axis("off")
                
#%%------FLATTEN, SHUFFLE, SPLIT AND RESIZE DATASETS----------------------

def get_dataset_partitions_tf(dataset, train_split=0.8):
        
    for image_batch, labels_batch in dataset:
        ndata    =  image_batch.shape[0]
        images   =  image_batch.numpy()/255  # Rescaling
        labels   =  labels_batch.numpy()
        # Flattening
        dataset  =  [images.reshape(ndata, -1), labels]
        
    ds_size = dataset[0].shape[0]
    
    train_size =  int(train_split * ds_size)
    train_ds   =  [dataset[0][:train_size,:], dataset[1][:train_size]]
    test_ds    =  [dataset[0][train_size:,:], dataset[1][train_size:]]
    plot_test  =  images[train_size:,:,:]
        
    return train_ds, test_ds, plot_test

train, test, plot_test = get_dataset_partitions_tf(dataset)

#%%------FITTING SVM MODEL----------------------
from sklearn.svm import SVC
model = SVC(kernel='linear', gamma='auto', probability=True)
model.fit(train[0], train[1])

#%%------PREDICT FROM MODEL----------------------
from sklearn.metrics import accuracy_score

def predict(model, test, class_names, T=1):

    predictions = model.predict_proba(test)
    predicted_class = model.predict(test)
    confidence = np.round(100 * (np.max(predictions, axis=1)), 2)

    return predicted_class, confidence
    
figsize  =  [5,6]
fontsize =  min(figsize)*2
plt.figure(figsize = figsize, dpi=200)

for i in range(4):
    ax = plt.subplot(2, 2, i + 1)
    plt.imshow(plot_test[i,:,:,0], cmap='gray')
    
    predicted_class, confidence = predict(model        =  model, 
                                          test         =  test[0][i,None],
                                          class_names  =  class_names)
    actual_class     =  class_names[test[1][i]]    
    predicted_class  =  class_names[predicted_class[0]]    
    string  =  "Actual: %s\n Predicted: %s\n Confidence: %4.2f " \
                %(actual_class, predicted_class, confidence) + '%'           
    # plt.text(x = 25, y = 240,
    #           s = string,
    #           fontsize=fontsize, color='white')
    plt.title(label = string, fontsize=fontsize, color='k')
    
    plt.axis("off")
        
plt.tight_layout()

print('Model Score : %5.2f ' %(accuracy_score(model.predict(test[0]),test[1])*100) + '%')
        