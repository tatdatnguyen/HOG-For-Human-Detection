# HOG METHOD BUILD FROM SCRATCH


## About HOG method 
HOG is an image feature descripts to describe the image based on the gradients directions and magnitudes. At the current time, this project supports calculating the following:
1. Horizontal and vertical gradients.
2. Gradient magnituge.
3. Gradient direction.
4. Histogram for a given cell.


## Installation
```bash
# clone project
git clone https://github.com/tatdatnguyen/HOG-For-Human-Detection.git

# create conda environment
conda create -n myenv python=3.7
conda activate myenv
pip install -r requirement.txt
```
## How to run
```bash
#run the human detection
python svm-train.py.py
```

```bash
#run the car logo classification
python train_logo_classification.py
```
 
 ```bash
#run the emotion recognition task
python src/train.py
python src/predict.py
```

## Goals

Three applications of Histogram of Oriented Gradients (HOG) feature extraction are employed:

1. Human Detection: Combines HOG with a linear Support Vector Machine (SVM) model for identifying human figures.
2. Car Logo Classification: Uses HOG features with a K-Nearest Neighbors (KNN) model to differentiate between various car logos.
3. Emotion Recognition: Applies HOG for capturing facial features to recognize different human emotions, though the specific classification model is not specified.

## The outcomes are notably favorable:

1. For the task of classifying humans, the results are commendable given the limited data, with a precision of 0.73, recall of 0.72, and an F1-score of 0.72.
2. In the classification of car logos, the performance is impressive, achieving a precision of 0.92, recall of 0.92, and an F1-score of 0.92.
3. The task of recognizing emotions shows exceptional results, with a precision of 0.98, recall of 0.97, and an F1-score of 0.97.

## About the modules
1. `HOG.py` -- This module contained how to extract HOG features of the training images.
2. `svm-train.py` -- This module is used to train the SVM classifier for human detection task.
3. `train_logo_classification.py` -- This module is used to train the KNN classifier for car logo detection.
4. `fer_bouchene.ipynb` --this module is used to train the emotion recognition model

## Contributors
- Nguyen Tat Dat - <nguyentatdat2811@gmai.com>
- Cuong Manh Quach - <quachmanhcuong03@gmail.com>
- Nguyen Tuan Anh  - <tuananhhmx4@gmail.com>
- Doan Minh Quan - <quandoan0902@gmail.com>

## Reference:
[Histogram of Oriented Gradients and Object Detection](http://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/)



