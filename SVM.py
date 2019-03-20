import os
import numpy as np
import cvxopt
import scipy.misc
import math
from matplotlib import pyplot
from sklearn.utils import shuffle
from sklearn import cross_validation


class Support_Vector_Machine():

    def __init__(self):
       
        self.c = 0.0
        self.lm = None
        self.sv_X = None
        self.sv_y = None
        self.b = None
        self.w = None

    def train(self, x, y):
        
        sample_len, feature_len = x.shape

       
        M = np.zeros((sample_len, sample_len))
        for i in range(sample_len):
            for j in range(sample_len):
                M[i, j] = np.dot(x[i], x[j])

       
        P = cvxopt.matrix(np.outer(y, y) * M)
        q = cvxopt.matrix(np.ones(sample_len) * -1)
        A = cvxopt.matrix(y, (1, sample_len), 'd')
        b = cvxopt.matrix(0.0)
        G = cvxopt.matrix(np.vstack((np.diag(np.ones(sample_len) * -1), np.identity(sample_len))))
        h = cvxopt.matrix(np.hstack((np.zeros(sample_len), np.ones(sample_len) * self.c)))

        
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        lm = np.ravel(sol['x'])

        
        y = np.asarray(y)
        sv = lm > 0.0e-7
        index = np.arange(len(lm))[sv]
        self.lm = lm[sv]
        self.sv_X = x[sv]
        self.sv_y = y[sv]

       
        self.b = 0.0
        for i in range(len(self.lm)):
            self.b = self.b + self.sv_y[i] - np.sum(self.lm * self.sv_y * M[index[i], sv])
        self.b /= len(self.lm)

        
        self.w = np.zeros(feature_len)
        for i in range(len(self.lm)):
            self.w += self.lm[i] * self.sv_y[i] * self.sv_X[i]

    def predict(self, x):
        return np.sign(np.dot(x, self.w) + self.b)


if __name__ == "__main__":
    def img_input(resize=False):
        X, y = [], []
        for path, direc, docs in os.walk("att_faces"):
            direc.sort()
          
            for subject in direc:
                files = os.listdir(path + '/' + subject)
                for file in files:
                    img = scipy.misc.imread(path + '/' + subject + '/' + file).astype(np.float32)
                    if resize:
                        img = scipy.misc.imresize(img, (56, 46)).astype(np.float32)
                    X.append(img.reshape(-1))
                    y.append(int(subject[1:]))
        X = np.asarray(X)
        y = np.asarray(y)
        return X, y


    def cross_val(X, y, DR=None, Alg=None, All=False):
        X, y = shuffle(X, y)
        kf = cross_validation.KFold(len(y), n_folds=5)
        
        average_accuracy = []
        
        fold = 1
        for train, test in kf:
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]



            if DR is None and Alg == "SVM" or All:
                print('\nRunning Fold', fold, 'for SVM')
               
                svm = Support_Vector_Machine()
                y_train_over = [None] * len(y_train)
                y_test_over = [None] * len(y_test)
                accuracies = 0
                print('Running SVM. wait...')
                for i in range(1, 41):
                    
                    for j in range(0, 320):
                        if y_train[j] == (i):
                            y_train_over[j] = 1
                        else:
                            y_train_over[j] = -1
                    for j in range(0, 80):
                        if y_test[j] == (i):
                            y_test_over[j] = 1
                        else:
                            y_test_over[j] = -1

                   
                    svm.train(X_train, y_train_over)
                    predict_class = svm.predict(X_test)
                    c = np.sum(predict_class == y_test_over)
                    accuracies += float(c) / len(predict_class) * 100
                accuracy=accuracies/40
                print('Accuracy: ', accuracy)
                average_accuracy.append(accuracy)
                if not All:
                    fold += 1



        if DR is None and Alg == "SVM" or All:
            print
            'Average accuracy: ', sum(average_accuracy) / 5.0, '\n'



    def task_selector():
        print("\n")
        X,y = img_input()
        cross_val(X,y,DR=None, Alg= "SVM")

    task_selector()
