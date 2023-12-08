import os
import glob
import cv2
from skimage import feature
import imutils
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from HOG import *


def _save(path, obj):
    with open(path, 'wb') as fn:
        pickle.dump(obj, fn)

def _preprocessing(fileType):
    data = []
    labels = []
    for path in glob.glob(fileType):
        # Extract the label from the parent directory name
        brand = os.path.basename(os.path.dirname(path))
        # Extract the file name
        fn = os.path.basename(path)

        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edged = imutils.auto_canny(gray)

        # Tìm contours trong edge map, chỉ giữ lại contours lớn nhất, được giả định là chứa logo xe.
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # Check if contours are found
        if cnts:
            c = max(cnts, key=cv2.contourArea)

            # Trích xuất logo của xe và resize lại kích thước ảnh logo về 200x200
            (x, y, w, h) = cv2.boundingRect(c)
            logo = gray[y:y + h, x:x + w]
            logo = cv2.resize(logo, (64, 128))

            # Khởi tạo HOG descriptor
            H = Hog_descriptor(logo, cell_size=16, bin_size=9)
            vector, image = H.extract()

            # update the data and labels
            data.append(vector)
            labels.append(brand)

    return data, labels

def _transform_data(data, labels):
    # Tạo input array X
    X = np.array(data)
    X = X.reshape(len(X), -1)
    # Tạo output array y
    le = LabelEncoder()
    le.fit(labels)
    y = le.transform(labels)
    y_ind = np.unique(y)
    y_dict = dict(zip(y_ind, le.classes_))
    return X, y, y_dict, le



#preprocessing the data
data, labels = _preprocessing('C:/Users/user/OneDrive/Desktop/HOG/CarLogo/TrainData/**/*.jpg')
dataTest, labelsTest = _preprocessing('C:/Users/user/OneDrive/Desktop/HOG/CarLogo/TestData/**/*.jpg')

print(len(labels))

_save('X_train.pkl', data)
_save('y_train.pkl', labels)

_save('X_test.pkl', dataTest)
_save('y_test.pkl', labelsTest)

X_train, y_train, y_dict, le = _transform_data(data, labels)

#import knn model
model = KNeighborsClassifier(n_neighbors = 1)
model.fit(X_train, y_train)

X_test, y_test, y_dict, le = _transform_data(dataTest, labelsTest)

y_predTest = model.predict(X_test)

# Kiểm tra độ chính xác của mô hình trên test
uniq_labels = list(y_dict.values())
print(classification_report(y_test, y_predTest, target_names = uniq_labels))