import numpy as np
import pandas as pd
class Star_Prediction:
    class MultiClassLogisticRegression:
        '''
          Using the trained model to predict the star rating of a new review
          Atrributes:
            review: a string of the review
            Returns:
                a float number of the predicted star rating
        '''
            
                
        def __init__(self, n_iter = 10000, thres=1e-3):

            '''
            Atrributes:
                n_iter: the number of iterations to train the model
                thres: the threshold to stop the training
            '''
            self.n_iter = n_iter
            self.thres = thres
        
        def fit(self, X, y, lr=0.01, rand_seed=4, verbose=False): 

            '''
                Parameters:
                    X: a numpy array of the features
                    y: a numpy array of the labels
                    lr: the learning rate
                    rand_seed: the random seed to initialize the weights
                    verbose: whether to print the training accuracy at each 100 iterations
                Returns:
                    None
            '''
            np.random.seed(rand_seed) 
            self.classes = np.unique(y)
            self.class_labels = {c:i for i,c in enumerate(self.classes)}
            X = self.add_bias(X)
            y = self.one_hot(y)
            self.loss = []
            self.acc = []
            self.weights = np.zeros(shape=(len(self.classes),X.shape[1]))
            self.fit_data(X, y,lr, verbose)
            return self
    
        def fit_data(self, X, y,lr, verbose):
            '''
                Fit data using gradient descent
                Parameters:
                    X: a numpy array of the features
                    y: a numpy array of the labels
                    lr: the learning rate
                    verbose: whether to print the training accuracy at each 100 iterations
                Returns:
                    None
            '''
            i = 0
            while (not self.n_iter or i < self.n_iter):
                self.loss.append(self.cross_entropy(y, self.predict_(X)))
                error = y - self.predict_(X)
                update = (lr * np.dot(error.T, X))
                self.weights += update
                if np.abs(update).max() < self.thres: break
                self.acc.append(self.evaluate_(X, y))
                if i % 100 == 0 and verbose: 
                    print('Training Accuray at {} iterations is {}'.format(i, self.evaluate_(X, y)))
                i +=1
                    
        
        def predict(self, X):
            '''
                Parameters:
                    X: a numpy array of the features
                Returns:
                    a numpy array of the predicted probabilities
            '''
            return self.predict_(self.add_bias(X))
        
        def predict_(self, X):
            '''
                Parameters:
                    X: a numpy array of the features
                Returns:
                    a numpy array of the predicted probabilities
            '''
            pre_vals = np.dot(X, self.weights.T).reshape(-1,len(self.classes))
            return self.softmax(pre_vals)
        
        def softmax(self, z):
            '''
                Parameters:
                    z: a numpy array of the pre-activation values
                Returns:
                    a numpy array of the softmax values
            '''
            return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1,1)

        def predict_classes(self, X):
            '''
                Parameters:
                    X: a numpy array of the features
                Returns:
                    a numpy array of the predicted classes
            '''
            self.probs_ = self.predict(X)
            return np.vectorize(lambda c: self.classes[c])(np.argmax(self.probs_, axis=1))
    
        def add_bias(self,X):
            '''
                Parameters:
                    X: a numpy array of the features
                Returns:
                    a numpy array of the features with bias
            '''
            return np.insert(X, 0, 1, axis=1)
    
        def get_randon_weights(self, row, col):
            '''
                Parameters:
                    row: the number of rows
                    col: the number of columns
                Returns:
                    a numpy array of the random weights
            '''
            return np.zeros(shape=(row,col))

        def one_hot(self, y):
            '''
                Parameters:
                    y: a numpy array of the labels
                Returns:
                    a numpy array of the one-hot encoded labels
            '''
            return np.eye(len(self.classes))[np.vectorize(lambda c: self.class_labels[c])(y).reshape(-1)]
        
        def score(self, X, y):
            '''
                Parameters:
                    X: a numpy array of the features
                    y: a numpy array of the labels
                Returns:
                    a float number of the accuracy
            '''
            return np.mean(self.predict_classes(X) == y)
        
        def evaluate_(self, X, y):
            '''
                Parameters:
                    X: a numpy array of the features
                    y: a numpy array of the labels
                Returns:
                    a float number of the accuracy
            '''
            return np.mean(np.argmax(self.predict_(X), axis=1) == np.argmax(y, axis=1))
        
        def cross_entropy(self, y, probs):
            '''
                Parameters:
                    y: a numpy array of the labels
                    probs: a numpy array of the predicted probabilities
                Returns:
                    a float number of the cross entropy
            '''
            return -1 * np.mean(y * np.log(probs))
    class LDA:
        '''
        Inverse transform

        Atributes:
        X_transformed: numpy array, shape = [n_samples,n_components]
        Returns:
        X_projected: numpy array, shape = [n_samples,n_features]
    
        '''
        def __init__(self,n_components):
            '''
            Parameters:
            n_components: int, default = 2

            '''
            self.n_components = n_components
            self.linear_discriminants = None
        def fit(self,X,y):

            '''
            Parameters:
            X: numpy array, shape = [n_samples,n_features]
            y: numpy array, shape = [n_samples,]
            '''
            n_features = X.shape[1]
            class_lables = np.unique(y)

            # S_W , S_B
            mean_overall = np.mean(X,axis = 0)
            S_W = np.zeros((n_features,n_features))
            S_B = np.zeros((n_features,n_features))

            for c in class_lables:
                X_c = X[y == c]
                mean_c = np.mean(X_c,axis = 0)
                # 4,4
                S_W += (X_c - mean_c).T.dot(X_c - mean_c)
                n_c = X_c.shape[0]
                mean_diff = (mean_c - mean_overall).reshape(n_features,1)
                S_B += n_c * (mean_diff).dot(mean_diff.T)
            A = np.linalg.inv(S_W).dot(S_B)
            # Same PCA
            eigenvalues,eigenveactors = np.linalg.eig(A)
            eigenveactors = eigenveactors.T
            indexs = np.argsort(abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[indexs]
            eigenvectors = eigenveactors[indexs]
            self.linear_discriminants = eigenvectors[0:self.n_components]

        def transform(self,X):

            '''
            Parameters:
            X: numpy array, shape = [n_samples,n_features]
            '''
            return np.dot(X,self.linear_discriminants.T)

    class Label_encoding:
        def __init__(self):
            '''
            Parameters:
            n_components: int, default = 2
            '''
            self.label_dict = {}
            self.label_dict_reverse = {}
        def fit(self,y):
            self.label_dict = {label:i for i,label in enumerate(np.unique(y))}
            self.label_dict_reverse = {i:label for i,label in enumerate(np.unique(y))}
        def transform(self,y):
            return np.vectorize(lambda c: self.label_dict[c])(y)
        def inverse_transform(self,y):
            return np.vectorize(lambda c: self.label_dict_reverse[c])(y)

    def fit(self,X,y):
        '''
        Function to fit the model. Labels are encoded to integers and then passed to the LDA model.
        Parameters:
        X: numpy array, shape = [n_samples,n_features]
        y: numpy array, shape = [n_samples,]

        '''
        self.dic_label_X = {}
        for col in X.columns:
            if X[col].dtype == 'object':
                le = self.Label_encoding()
                le.fit(X[col])
                X[col] = le.transform(X[col])
                self.dic_label_X[col] = le
        # Label y
        self.label_y = self.Label_encoding()
        self.label_y.fit(y)
        self.y = self.label_y.transform(y)
        self.LDA = self.LDA(4)
        self.LDA.fit(X.values,self.y)
        self.X = self.LDA.transform(X.values)
        self.model = self.MultiClassLogisticRegression()
        self.model.fit(self.X,self.y)
    
    def predict(self,X):
        for col in X.columns:
            if col in self.dic_label_X:
                X[col] = self.dic_label_X[col].transform(X[col])
        X = self.LDA.transform(X.values)
        y = self.model.predict_classes(X)
        return self.label_y.inverse_transform(y)