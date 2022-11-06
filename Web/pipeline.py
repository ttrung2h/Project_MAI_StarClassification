import numpy as np
import pandas as pd
class Star_Prediction:
    class MultiClassLogisticRegression:
        
        def __init__(self, n_iter = 10000, thres=1e-3):
            self.n_iter = n_iter
            self.thres = thres
        
        def fit(self, X, y, batch_size=64, lr=0.001, rand_seed=4, verbose=False): 
            np.random.seed(rand_seed) 
            self.classes = np.unique(y)
            self.class_labels = {c:i for i,c in enumerate(self.classes)}
            X = self.add_bias(X)
            y = self.one_hot(y)
            self.loss = []
            self.weights = np.zeros(shape=(len(self.classes),X.shape[1]))
            self.fit_data(X, y, batch_size, lr, verbose)
            return self
    
        def fit_data(self, X, y, batch_size, lr, verbose):
            i = 0
            while (not self.n_iter or i < self.n_iter):
                self.loss.append(self.cross_entropy(y, self.predict_(X)))
                idx = np.random.choice(X.shape[0], batch_size)
                X_batch, y_batch = X[idx], y[idx]
                error = y_batch - self.predict_(X_batch)
                update = (lr * np.dot(error.T, X_batch))
                self.weights += update
                if np.abs(update).max() < self.thres: break
                
                if i % 1000 == 0 and verbose: 
                    print(X.shape)
                    print(y)
                    print(' Training Accuray at {} iterations is {}'.format(i, self.evaluate_(X, y)))
                i +=1
        
        def predict(self, X):
            return self.predict_(self.add_bias(X))
        
        def predict_(self, X):
            pre_vals = np.dot(X, self.weights.T).reshape(-1,len(self.classes))
            return self.softmax(pre_vals)
        
        def softmax(self, z):
            return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1,1)

        def predict_classes(self, X):
            self.probs_ = self.predict(X)
            return np.vectorize(lambda c: self.classes[c])(np.argmax(self.probs_, axis=1))
    
        def add_bias(self,X):
            return np.insert(X, 0, 1, axis=1)
    
        def get_randon_weights(self, row, col):
            return np.zeros(shape=(row,col))

        def one_hot(self, y):
            return np.eye(len(self.classes))[np.vectorize(lambda c: self.class_labels[c])(y).reshape(-1)]
        
        def score(self, X, y):
            return np.mean(self.predict_classes(X) == y)
        
        def evaluate_(self, X, y):
            return np.mean(np.argmax(self.predict_(X), axis=1) == np.argmax(y, axis=1))
        
        def cross_entropy(self, y, probs):
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