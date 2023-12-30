from train import print_score, X_train, y_train, X_test, y_test
import pickle

# load

with open('./model.pkl', 'rb') as f:
    svm_clf = pickle.load(f)
print_score(svm_clf, X_train, y_train, X_test, y_test, train=False)