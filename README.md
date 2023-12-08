# HOG-method-build-from-scratch


HOG is an image feature descripts to describe the image based on the gradients directions and magnitudes. At the current time, this project supports calculating the following:
1. Horizontal and vertical gradients.
2. Gradient magnituge.
3. Gradient direction.
4. Histogram for a given cell.

Requirements: Python >= 3.7

We applied HOG method in 2 tasks which are:
1. Human Detection (HOG features extraction combined with linear SVM model)
2. Car Logo Classification (HOG features extraction combined with KNN model)

The results are quite positive with: 
1. accuracy 0.7-0.75 for the human detection task (small amount of data)
2. accuracy 0.8-0.9 for the car logo classification task

About the modules

1. `HOG.py` -- This module contained how to extract HOG features of the training images.
2. `svm-train.py` -- This module is used to train the SVM classifier for human detection task.
3. `train_logo_classification.py` -- This module is used to train the KNN classifier for car logo detection.

Reference:
[Histogram of Oriented Gradients and Object Detection](http://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/)



