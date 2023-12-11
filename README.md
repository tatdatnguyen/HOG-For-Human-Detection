# HOG METHOD BUILD FROM SCRATCH


## About HOG method 
HOG is an image feature descripts to describe the image based on the gradients directions and magnitudes. At the current time, this project supports calculating the following:
1. Horizontal and vertical gradients.
2. Gradient magnituge.
3. Gradient direction.
4. Histogram for a given cell.

## Requirements
Python >= 3.7

## Usage
```bash
#run the human detection
python svm-train.py.py
```

```bash
#run the car logo classification
python train_logo_classification.py
```

## We applied HOG method in 3 tasks
1. Human Detection (HOG features extraction combined with linear SVM model)
2. Car Logo Classification (HOG features extraction combined with KNN model)
3. Emotion Recognition 

## The results are quite positive 
1. Accuracy 0.7-0.75 for the human detection task (small amount of data)
2. Accuracy 0.8-0.9 for the car logo classification task
3. Accuracy 0.8-0.9

## About the modules
1. `HOG.py` -- This module contained how to extract HOG features of the training images.
2. `svm-train.py` -- This module is used to train the SVM classifier for human detection task.
3. `train_logo_classification.py` -- This module is used to train the KNN classifier for car logo detection.
4. `fer_bouchene.ipynb` --this module is used to train the emotion recognition model

## Reference:
[Histogram of Oriented Gradients and Object Detection](http://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/)



