from HOG import *
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import glob,os
from sklearn import metrics
from sklearn.metrics import classification_report 
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt


X = []
Y = []

pos_im_path = 'C:/Users/user/OneDrive/Desktop/HOG/img/1'
neg_im_path = 'C:/Users/user/OneDrive/Desktop/HOG/img/0'

# Load the positive features
for filename in glob.glob(os.path.join(pos_im_path,"*.png")):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(64,128))
    hog = Hog_descriptor(img, cell_size=16, bin_size=9)
    vector, image = hog.extract()
    X.append(vector)
    Y.append(1)

# Load the negative features
for filename in glob.glob(os.path.join(neg_im_path,"*.png")):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(64,128))
    hog = Hog_descriptor(img, cell_size=16, bin_size=9)
    vector, image = hog.extract()
    X.append(vector)
    Y.append(0)
    
X = np.float32(X)
X = X.reshape(len(X), -1)
Y = np.array(Y)

#split train/test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

print('Train Data:',len(X_train))
print('Train Labels (1,0)',len(y_train))

#train
model = LinearSVC()
model.fit(X_train,y_train)

# predict
y_pred = model.predict(X_test)

# confusion matrix and accuracy
print(f"Classification report for classifier {model}:\n"
      f"{metrics.classification_report(y_test, y_pred)}\n")