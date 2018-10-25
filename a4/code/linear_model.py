import numpy as np
from numpy.linalg import solve
from numpy import linalg
import findMin
from scipy.optimize import approx_fprime
import utils

class logReg:
    # Logistic Regression
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)



class logRegL0(logReg):
    # L0 Regularized Logistic Regression
    def __init__(self, L0_lambda=1.0, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.L0_lambda = L0_lambda
        self.maxEvals = maxEvals

    def fit(self, X, y):
        n, d = X.shape
        minimize = lambda ind: findMin.findMin(self.funObj,
                                                  np.zeros(len(ind)),
                                                  self.maxEvals,
                                                  X[:, ind], y, verbose=0)
        selected = set()
        selected.add(0)
        minLoss = np.inf
        oldLoss = 0
        bestFeature = -1

        while minLoss != oldLoss:

            oldLoss = minLoss
            print("Epoch %d " % len(selected))
            print("Selected feature: %d" % (bestFeature))
            print("Min Loss: %.3f\n" % minLoss)
            bestScore = minLoss
            for i in range(d):
                if i in selected:
                    continue

                selected_new = selected | {i}
                # TODO for Q2.3: Fit the model with 'i' added to the features,
                # then compute the loss and update the minLoss/bestFeature


                cur_w, _ =minimize(list(selected_new));
                y_hat = np.sign(X[:,list(selected_new)]@cur_w)
                cur_score = np.mean(y != y_hat) + self.L0_lambda*len(selected_new)

                if cur_score < bestScore:
                    bestScore = cur_score
                    bestFeature = i

            minLoss = bestScore

            selected.add(bestFeature)

        self.w = np.zeros(d)
        self.w[list(selected)], _ = minimize(list(selected))


class leastSquaresClassifier:
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            self.W[i] = np.linalg.solve(X.T@X+0.0001*np.eye(d), X.T@ytmp)

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)


class logRegL2(logReg):
    def __init__(self, lammy=1.0, verbose=0, maxEvals=400):
        self.verbose = verbose
        self.L2_lambda = lammy
        self.maxEvals = maxEvals

    def fun_grad(self, w, X, y):

        #calculate the function value
        yXw = y* X.dot(w)
        f = np.sum(np.log(1.+ np.exp(-yXw))) + (self.L2_lambda/2)*linalg.norm(w)**2

        #calculate the gradient
        exp = np.exp(-yXw)
        res = (-y* exp) / ((1.+exp))
        g = X.T .dot(res) + self.L2_lambda * w

        return f,g



    def fit(self, X, y):

        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.fun_grad, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)

    def predict(self, X):
        return np.sign(X@self.w)


class logRegL1(logReg):
    def __init__(self,L1_lambda=1.0, verbose=0, maxEvals=400):
        self.verbose = verbose
        self.L1_lambda = L1_lambda
        self.maxEvals = maxEvals

    def fun_grad(self, w, X, y):

        #calculate the function value
        yXw = y*X.dot(w)
        f = np.sum (np.log(1.+ np.exp(-yXw)))

        #calculate the gradient
        exp = np.exp(-yXw)
        res = (-y * exp) / (1.+exp)
        g = X.T.dot(res)

        return f, g



    def fit(self, X, y):

        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMinL1(self.fun_grad, self.w, self.L1_lambda,
                                      self.maxEvals, X, y, verbose=self.verbose)

    def predict(self, X):
        return np.sign(X@self.w)


class logLinearClassifier:

    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit (self, X,y):

        n, d = X.shape
        #number of labels
        k = np.unique(y).size
        self.w = np.zeros((k,d))

        for i in range(k):

            #create the new label
            new_y = np.zeros(y.size)
            for j in range (y.size):
                if y[j] == i:
                    new_y[j] = 1
                else:
                    new_y[j] = -1


            #utils.check_gradient(self, X, new_y)
            self.w[i,:],_ = findMin.findMin(self.funObj, self.w[i,:],
                                      self.maxEvals, X, new_y, verbose=self.verbose)




    def predict (self,X):

        n, d = X.shape
        y = np.zeros(n)

        xw = X@self.w.T

        for i in range(n):
            y[i] = np.argmax(xw[i,:])

        return y



class softmaxClassifier:
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def indicator (self,y, c):

        i = np.array(np.array(y==c)*1)

        return i


    def funObj(self, w, X, y):

        n, d = X.shape
        k = np.unique(y).size
        W = np. reshape(w,(k,d))
        loss = 0


        for i in range(n):

            loss += -W[y[i],:].dot(X[i,:])+ np.log(np.sum(np.exp(W*X[i,:].T)))

        f = loss


        # k is the number of labels
        k = np.unique(y).size
        g = np.zeros((k, d))

        s = np.sum(np.exp(X@W.T), axis = 1)
        for c in range(k):

            res = np.exp(X@W[c,:])/s - self.indicator(y,c)
            res = np.multiply(X, res.reshape(n,1))
            g[c] = np.sum(res, axis = 0)

        #print(g)

        return f,g.flatten()

    def fit (self, X, y ):

        n ,d = X.shape
        k = np.unique(y).size

        w = np.zeros((k,d))
        self.w = w.flatten()
        #utils.check_gradient(self, X, y)

        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)

        self.w = np.reshape(self.w,(k,d))

    def predict (self, X):
        return np.argmax(X@self.w.T, axis=1)


