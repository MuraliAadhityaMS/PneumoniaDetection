#%% ------IMPORTS---------------------
"""
Created on Tue Nov 22 09:44:02 2022
@author: Adi
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

#%% ------LOAD DATASETS----------------------

BATCH_SIZE   =   500
IMAGE_SIZE   =   256
CHANNELS     =   1 # grayscale 
FOLDER       =   './Data/for_test/500'

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
# We are shuffling here to plot datasets in the next cell so that all cases can be seen

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
        images   =  image_batch.numpy()/255     # Rescaling
        labels   =  labels_batch.numpy()        # Flattening
        dataset  =  [images.reshape(ndata, -1), labels]
        
    ds_size = dataset[0].shape[0]
    
    train_size =  int(train_split * ds_size)
    train_ds   =  [dataset[0][:train_size,:], dataset[1][:train_size]]
    test_ds    =  [dataset[0][train_size:,:], dataset[1][train_size:]]
    plot_test  =  images[train_size:,:,:]
        
    return train_ds, test_ds, plot_test, dataset

train, test, plot_test, list_dataset = get_dataset_partitions_tf(dataset)

#%%------FIND VALIDATION SCORE----------------------
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve

param_range = np.logspace(-4, 1, 10)

train_scores, test_scores = validation_curve(
    SVC(kernel='poly', degree=3),
    list_dataset[0],
    list_dataset[1],
    param_name="C",
    param_range=param_range,
    scoring="accuracy",
    n_jobs=2,
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

#%%------PLOT VALIDATION CURVE----------------------
figsize  =  [6,5]
fontsize =  min(figsize)*4
plt.figure(figsize = figsize, dpi=200)

# plt.title("Validation Curve with SVM", fontsize=fontsize)
# plt.xlabel(r"Margin ($\gamma$)", fontsize=fontsize)
plt.xlabel('Reg. Param (C)', fontsize=fontsize)
plt.ylabel("Score", fontsize=fontsize)
# plt.ylim(0, 1)
plt.xlim(1e-4, 1e1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training", 
         color="darkorange", lw=lw)

plt.fill_between(
    param_range,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.2,
    color="darkorange",
    lw=lw,
)
plt.semilogx(
    param_range, test_scores_mean, label="Test", color="navy", lw=lw
)
plt.fill_between(
    param_range,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.2,
    color="navy",
    lw=lw,
)

plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(loc="best", fontsize=fontsize)
plt.show()

#%%------FITTING SVM MODEL----------------------
from sklearn.svm import SVC
model = SVC(kernel='linear', probability=True, C=2e-4, tol=1e-5)

start_time = time.time()
model.fit(train[0], train[1])
print("--- %s seconds ---" % (time.time() - start_time))

if model.fit_status_ == 0:
    print('SVM MODEL HAS CONVERGED')
else :
    print('SVM MODEL HAS NOT CONVERGED')

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

test_shape  =  test[0].shape[0]
random      =  np.random.randint(low=0, high=test_shape, size=4, dtype=int)

for j in range(random.shape[0]) :
    ax = plt.subplot(2, 2, j + 1)
    i  =  random[j]
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
        
#%%------PLOTTING SVM MARGIN----------------------

figsize  =  [5,5]
fontsize =  min(figsize)*4
m_size   =  min(figsize)*16
lw       =  min(figsize)*2
plt.figure(figsize = figsize, dpi=200)

feature = [2046, 2302]      # Between top 2 weights
# feature = [2046, 1640]    # Between highest and lowest

# color   =  model.predict(list_dataset[0])
color   =  list_dataset[1]
color   =  np.array(color, dtype=str)
color[color == '0'] = 'b'
color[color == '1'] = 'r'
        
plt.scatter(list_dataset[0][:,feature[0]], list_dataset[0][:,feature[1]], marker="s", 
            c=color, edgecolor='w', s = m_size)

w = model.coef_[0]           
b = model.intercept_[0]
x_points = np.linspace(0, 1)
y_points =  -(w[feature[0]] * x_points + b) / w[feature[1]] 

margin = 2 * np.sqrt(np.sum(w ** 2))
new_points_up   = y_points + margin/2
new_points_down = y_points - margin/2

plt.plot(x_points, y_points, c='k', lw=3)

# Plot Margin
plt.fill_between(
    x_points,
    new_points_down,
    new_points_up,
    alpha=0.3,
    color="lime",
    lw=lw,
)

plt.xlim([0,1])
plt.ylim([0,1])
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('Feature (%d)' %(feature[0]), fontsize=fontsize)
plt.ylabel('Feature (%d)' %(feature[1]), fontsize=fontsize)

from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='s', color='b', label='Scatter', linestyle='None',
                          markeredgecolor='w', markersize= m_size*0.12),
                   Line2D([0], [0], marker='s', color='r', label='Scatter', linestyle='None',
                          markeredgecolor='w', markersize= m_size*0.12)]
                   # Line2D([0], [0], color='k', lw=4, label='Line')]


plt.legend(handles = legend_elements,
            labels = [class_names[0], class_names[1]],#, 'Fit'], 
            loc="upper left", fontsize=fontsize*0.7)
plt.show()
