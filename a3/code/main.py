import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

import linear_model
import utils

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

def test_and_plot(model,X,y,Xtest=None,ytest=None,title=None,filename=None):

    # Compute training error
    yhat = model.predict(X)
    trainError = np.mean((yhat - y)**2)
    print("Training error = %.1f" % trainError)

    # Compute test error
    if Xtest is not None and ytest is not None:
        yhat = model.predict(Xtest)
        testError = np.mean((yhat - ytest)**2)
        print("Test error     = %.1f" % testError)

    # Plot model
    plt.figure()
    plt.plot(X,y,'b.')

    # Choose points to evaluate the function
    Xgrid = np.linspace(np.min(X),np.max(X),1000)[:,None]
    ygrid = model.predict(Xgrid)
    plt.plot(Xgrid, ygrid, 'g')

    if title is not None:
        plt.title(title)

    if filename is not None:
        filename = os.path.join("..", "figs", filename)
        print("Saving", filename)
        plt.savefig(filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)
    io_args = parser.parse_args()
    question = io_args.question


    if question == "2":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LeastSquares()
        model.fit(X,y)
        print(model.w)

        test_and_plot(model,X,y,title="Least Squares",filename="least_squares_outliers.pdf")

    elif question == "2.1":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        ''' YOUR CODE HERE '''
        model = linear_model.WeightedLeastSquares()
        z = np.ones(500)
        z[400:] = 0.1
        z = np.diag(z)
        model.fit(X,y,z)

        test_and_plot(model,X,y,title="Weighted Least Squares",filename="least_squares_outliers_weighted.pdf")

    elif question == "2.3":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        model = linear_model.LinearModelGradient()
        model.fit(X,y)
        print(model.w)

        test_and_plot(model,X,y,title="Robust (L1) Linear Regression",filename="least_squares_robust.pdf")

    elif question == "3":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # Fit least-squares model
        model = linear_model.LeastSquares()
        model.fit(X,y)

        test_and_plot(model,X,y,Xtest,ytest,title="Least Squares, no bias",filename="least_squares_no_bias.pdf")

    elif question == "3.1":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        ''' YOUR CODE HERE'''
        model = linear_model.LeastSquaresBias()
        model.fit(X,y)

        test_and_plot(model,X,y,Xtest,ytest,title="Least Squares, with bias",filename="least_squares_bias.pdf")

    elif question == "3.2":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        for p in range(11):
            print("p = %d" % p)

            ''' YOUR CODE HERE '''
            model = linear_model.LeastSquaresPoly(p)
            model.fit(X,y)

            test_and_plot(model,X,y,Xtest,ytest,title='Least Squares Polynomial p = %d'%p,filename="PolyBasis%d.pdf"%p)


    elif question == "4":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        n,d = X.shape
        y_n, y_d = y.shape


        # Split training data into a training and a validation set


        Xtrain = X[0:n//2]
        ytrain = y[0:n//2]
        Xvalid = X[n//2: n]
        yvalid = y[n//2: n]


        #Xtrain, Xvalid, ytrain, yvalid = train_test_split(X,y,train_size = 0.5)

        # Find best value of RBF kernel parameter,
        # training on the train set and validating on the validation set




        minErr = np.inf
        for s in range(-15,16):
            sigma = 2 ** s

            # Train on the training set
            model = linear_model.LeastSquaresRBF(sigma)



            model.fit(Xtrain,ytrain)

            # Compute the error on the validation set
            yhat = model.predict(Xvalid)
            validError = np.mean((yhat - yvalid)**2)
            print("Error with sigma = 2^%-3d = %.1f" % (s ,validError))

            # Keep track of the lowest validation error
            if validError < minErr:
                minErr = validError
                bestSigma = sigma

        print("Value of sigma that achieved the lowest validation error =", bestSigma)

        # Now fit the model based on the full dataset.
        print("Refitting on full training set...\n")
        model = linear_model.LeastSquaresRBF(bestSigma)
        model.fit(X,y)

        test_and_plot(model,X,y,Xtest,ytest,
            title="Least Squares with RBF kernel and $\sigma={}$".format(bestSigma),
            filename="least_squares_rbf_bad.pdf")


    elif question == "4.1":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        n,d = X.shape

        # Split training data into a training and a validation set

        '''
        Xtrain = X[0:n//2]
        ytrain = y[0:n//2]
        Xvalid = X[n//2: n]
        yvalid = y[n//2: n]
        '''

        Xtrain, Xvalid, ytrain, yvalid = train_test_split(X,y,train_size = 0.5)

        # Find best value of RBF kernel parameter,
        # training on the train set and validating on the validation set




        minErr = np.inf
        for s in range(-15,16):
            sigma = 2 ** s

            # Train on the training set
            model = linear_model.LeastSquaresRBF(sigma)



            model.fit(Xtrain,ytrain)

            # Compute the error on the validation set
            yhat = model.predict(Xvalid)
            validError = np.mean((yhat - yvalid)**2)
            print("Error with sigma = 2^%-3d = %.1f" % (s ,validError))

            # Keep track of the lowest validation error
            if validError < minErr:
                minErr = validError
                bestSigma = sigma

        print("Value of sigma that achieved the lowest validation error =", bestSigma)

        # Now fit the model based on the full dataset.
        print("Refitting on full training set...\n")
        model = linear_model.LeastSquaresRBF(bestSigma)
        model.fit(X,y)

        test_and_plot(model,X,y,Xtest,ytest,
            title="Least Squares with RBF kernel and $\sigma={}$".format(bestSigma),
            filename="least_squares_rbf_random.pdf")

    elif question == "4.2":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        n,d = X.shape
        y_n, y_d = y.shape

        Z = np.column_stack((X,y))
        np.random.shuffle(Z)
        X = Z[:,0].reshape((n,d))
        y=  Z[:,1].reshape((y_n,y_d))



        minErr = np.inf
        slice_sz = int(n/10)
        for s in range(-15,16):
            sigma = 2 ** s

            # Train on the training set
            model = linear_model.LeastSquaresRBF(sigma)

            #cross validation
            error_sum = 0;
            for i in range(10):
                Xvalid = X[i*slice_sz:(i+1)*slice_sz,:]
                yvalid = y[i*slice_sz:(i+1)*slice_sz,:]

                Xtrain = np.concatenate((X[0:i*slice_sz,:],X[(i+1)*slice_sz:,:]))
                ytrain = np.concatenate((y[0:i*slice_sz,:],y[(i+1)*slice_sz:,:]))

                model.fit(Xtrain,ytrain)
                yhat = model.predict(Xvalid)
                error_sum += np.mean((yhat - yvalid)**2)

            validError = error_sum / 10

            #validError = np.mean((yhat - yvalid)**2)
            print("Error with sigma = 2^%-3d = %.1f" % (s ,validError))

            # Keep track of the lowest validation error
            if validError < minErr:
                minErr = validError
                bestSigma = sigma

        print("Value of sigma that achieved the lowest validation error =", bestSigma)

        # Now fit the model based on the full dataset.
        print("Refitting on full training set...\n")
        model = linear_model.LeastSquaresRBF(bestSigma)
        model.fit(X,y)

        test_and_plot(model,X,y,Xtest,ytest,
            title="Least Squares with RBF kernel and $\sigma={}$".format(bestSigma),
            filename="least_squares_rbf_cross.pdf")

        #sklearn cross val

        minErr = np.inf
        for s in range(-15,16):

            sigma = 2 ** s

            model = linear_model.LeastSquaresRBF(sigma)

            validError=np.min(cross_val_score(model,X,y, cv = 10))

            if validError < minErr:
                minErr = validError
                bestSigma = sigma

        print("Value of sigma that achieved the lowest validation error in cross_val_score =", bestSigma)

        # Now fit the model based on the full dataset.
        print("Refitting on full training set...\n")
        model = linear_model.LeastSquaresRBF(bestSigma)
        model.fit(X,y)

        test_and_plot(model,X,y,Xtest,ytest,
            title="Least Squares with RBF kernel and $\sigma={}$".format(bestSigma),
            filename="least_squares_rbf_cross_val_score.pdf")




