# standard Python imports
import os
import argparse
import time
import pickle

# 3rd party libraries
import numpy as np                              # this comes with Anaconda
import matplotlib.pyplot as plt                 # this comes with Anaconda
import pandas as pd                             # this comes with Anaconda
from sklearn.tree import DecisionTreeClassifier # see http://scikit-learn.org/stable/install.html
from sklearn.neighbors import KNeighborsClassifier # same as above

# CPSC 340 code
import utils
from decision_stump import DecisionStump, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from knn import KNN, CNN

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True,
        choices=["1.1", "2", "2.2", "2.3", "2.4", "3", "3.1", "3.2", "4.1", "4.2", "5"])

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1.1":
        # Load the fluTrends dataset
        df = pd.read_csv(os.path.join('..','data','fluTrends.csv'))
        X = df.values
        names = df.columns.values

        ''' YOUR CODE HERE FOR Q1.1 '''
        percentlist=[0.05,0.25,0.5,0.75,0.95]
        quantileList=[np.percentile(X,5),np.percentile(X,25),np.percentile(X,50),np.percentile(X,75),np.percentile(X,95)]
        for x,y in zip(percentlist,quantileList):
            print('{} quantile :{:.4}'.format(int(x*100),y))

        print('max: {:.4}'.format(np.max(X)))
        print('min: {:.4}'.format(np.min(X)))
        print('mean: {:.4}'.format(np.mean(X)))
        print('median: {:.4}'.format(np.median(X)))
        print('mode: {:.4}'.format(utils.mode(X)))

        meanList=[]
        varianceList=[]
        for index, col in df.iteritems():
            meanList.append(col.values.mean())
            varianceList.append(col.values.var())


        print('highest mean region: %s'%df.columns[meanList.index(max(meanList))])
        print('lowest mean region: %s'%df.columns[meanList.index(min(meanList))])
        print('highest variance region: %s'%df.columns[varianceList.index(max(varianceList))])
        print('lowest variance region: %s'%df.columns[varianceList.index(min(varianceList))])
        pass

    elif question == "2":

        # 1. Load citiesSmall dataset
        dataset = load_dataset("citiesSmall.pkl")
        X = dataset["X"]
        y = dataset["y"]

        print(X[0:5])
        # 2. Evaluate majority predictor model
        y_pred = np.zeros(y.size) + utils.mode(y)

        error = np.mean(y_pred != y)
        print("Mode predictor error: %.3f" % error)

        # 3. Evaluate decision stump
        model = DecisionStumpEquality()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y)
        print("Decision Stump with inequality rule error: %.3f" % error)

        # PLOT RESULT
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q2_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    elif question == "2.2":
        # 1. Load citiesSmall dataset
        dataset = load_dataset("citiesSmall.pkl")
        X = dataset["X"]
        y = dataset["y"]

        # 3. Evaluate decision stump
        model = DecisionStump()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y)
        print("Decision Stump with inequality rule error: %.3f" % error)

        # PLOT RESULT
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q2_2_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == "2.3":
        # 1. Load citiesSmall dataset
        dataset = load_dataset("citiesSmall.pkl")
        X = dataset["X"]
        y = dataset["y"]

        # 2. Evaluate decision tree
        model = DecisionTree(max_depth=2)
        model.fit(X, y)

        y_pred = model.predict(X)
        error = np.mean(y_pred != y)

        print("Error: %.3f" % error)

    elif question == "2.4":
        dataset = load_dataset("citiesSmall.pkl")
        X = dataset["X"]
        y = dataset["y"]
        print("n = %d" % X.shape[0])

        depths = np.arange(1,15) # depths to try

        t = time.time()
        my_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTree(max_depth=max_depth)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors[i] = np.mean(y_pred != y)
        print("Our decision tree took %f seconds" % (time.time()-t))

        plt.plot(depths, my_tree_errors, label="mine")

        t = time.time()
        sklearn_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors[i] = np.mean(y_pred != y)
        print("scikit-learn's decision tree took %f seconds" % (time.time()-t))


        plt.plot(depths, my_tree_errors, label="sklearn")
        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "q2_4_tree_errors.pdf")
        plt.savefig(fname)

        tree = DecisionTreeClassifier(max_depth=1)
        tree.fit(X, y)

        y_pred = tree.predict(X)
        error = np.mean(y_pred != y)

        print("Error: %.3f" % error)


    elif question == "3":
        dataset = load_dataset("citiesSmall.pkl")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]
        model = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=1)
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)

    elif question == "3.1":

        dataset = load_dataset("citiesSmall.pkl")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]

        tr_error = np.zeros(15)
        te_error = np.zeros(15)
        depths = np.arange(1,16)

        for d in depths:
            model = DecisionTreeClassifier(max_depth=d, criterion='entropy',
                        random_state=1)
            model.fit(X,y)

            y_pred = model.predict(X)
            tr_error[d-1] =  np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error[d-1] = np.mean(y_pred != y_test)


        plt.plot(depths, tr_error, label="tr_error")
        plt.plot(depths, te_error, label="te_error")

        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        plt.title('Testing error and Training error vs depth')
        fname = os.path.join("..", "figs", "q3_1_varing_depths.pdf")
        plt.savefig(fname)


        pass

    elif question == "3.2":
        dataset = load_dataset("citiesSmall.pkl")
        X_total, y_total = dataset["X"], dataset["y"]
        N, D= X_total.shape
        X=X_total[:int(N/2),:]
        y=y_total[:int(N/2)]
        X_test, y_test = X_total[int(N/2):,:] , y_total[int(N/2):]

        minValError = N/2
        depth1 = None
        depth2 = None

        for d in range (1,16):
            model = DecisionTreeClassifier(max_depth=d, criterion='entropy',
                        random_state=1)
            model.fit(X,y)

            y_pred = model.predict(X_test)
            error = np.sum(y_pred != y_test)

            if error < minValError:
                minValError = error
                depth1 = d


        minValError = N/2
        for d in range (1,16):
            model = DecisionTreeClassifier(max_depth=d, criterion='entropy',
                        random_state=1)
            model.fit(X_test,y_test)

            y_pred = model.predict(X)
            error = np.sum(y_pred != y)

            if error < minValError:
                minValError = error
                depth2 = d

        print( 'depth1 is {}'.format(depth1))
        print( 'depth2 is {}'.format(depth2))




        pass

    if question == '4.1':

        dataset = load_dataset("citiesSmall.pkl")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]


        l = [1,3,10]
        tr_error=[]
        te_error=[]

        for k in l:
            model = KNN(k)
            model.fit(X,y)

            y_pred = model.predict(X)
            tr_error.append( np.mean(y_pred != y) )

            y_pred = model.predict(X_test)
            te_error.append( np.mean(y_pred != y_test) )

        print('Training error: ')
        for i in range(3):
            print('{}:{}'.format(l[i],tr_error[i]))

        print('Test error: ')
        for i in range(3):
            print('{}:{}'.format(l[i],te_error[i]))

        #Generate the plot, not useful otherwise
        #knn_model = KNN(1)
        #scikit_model = KNeighborsClassifier(n_neighbors=1)

        #knn_model.fit(X,y)
        #scikit_model.fit(X,y)
        #utils.plotClassifier(knn_model, X, y)

        pass

    if question == '4.2':
        dataset = load_dataset("citiesBig1.pkl")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]

        #4.2.1
        '''
        cnn_model = CNN(1)
        cnn_model.fit(X,y)

        knn_model = KNN(1)
        knn_model.fit(X,y)
        start = time.time()
        cnn_model.predict (X_test)
        print('cnn prediction time : %.3f'%(time.time()-start))

        start = time.time()
        knn_model.predict (X_test)
        print('knn prediction time : %.3f'%(time.time()-start))
        '''


        cnn_model = CNN(1)
        cnn_model.fit(X,y)

        #4.2.2
        '''
        y_pred = cnn_model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = cnn_model.predict(X_test)
        te_error = np.mean(y_pred != y_test)

        num_var ,_= cnn_model.X.shape
        print(' training error is {:.3}, test error is {:.3}, number of variable is {}'.format(tr_error,te_error,num_var))
        '''

        #4.2.3
        '''
        utils.plotClassifier(cnn_model,X,y)
        '''

        #4.2.6
        '''
        dataset = load_dataset("citiesBig2.pkl")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]
        cnn_model = CNN(1)
        cnn_model.fit(X,y)
        y_pred = cnn_model.predict(X)
        tr_error = np.mean(y_pred != y)
        y_pred = cnn_model.predict(X_test)
        te_error = np.mean(y_pred != y_test)

        num_var ,_= cnn_model.X.shape
        print(' training error is {:.3}, test error is {:.3}, number of variable is {}'.format(tr_error,te_error,num_var))
        '''

        #4.2.7
        model = DecisionTreeClassifier()
        model.fit(X,y)

        utils.plotClassifier(model, X,y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)

        print(' training error is {:.3}, test error is {:.3}'.format(tr_error,te_error))



        pass
