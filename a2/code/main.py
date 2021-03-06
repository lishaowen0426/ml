# basics
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time

# sklearn imports
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from skimage.io import imread, imshow, imsave


# our code
from naive_bayes import NaiveBayes

from decision_stump import DecisionStump, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from random_tree import RandomTree
from random_forest import RandomForest # TODO

from kmeans import Kmeans
# from kmedians import Kmedians # TODO
from quantize_image import ImageQuantizer
from sklearn.cluster import DBSCAN

from kmedians import Kmedians

def plot_2dclustering(X,y):
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.title('Cluster Plot')


def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == '1.2':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]
        groupnames = dataset["groupnames"]
        wordlist = dataset["wordlist"]

        # TODO: your code here
        #1.1
        print('question 1.1: {}'.format(wordlist[49]))

        #1.2
        N,D = X.shape
        print( 'question 1.2: {}'.format(wordlist[X[499,:] == 1]))

        #1.3
        print( 'question 1.3: {}'.format(groupnames[y[499]]))


    elif question == '1.3':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]

        print("d = %d" % X.shape[1])
        print("n = %d" % X.shape[0])
        print("t = %d" % X_valid.shape[0])
        print("Num classes = %d" % len(np.unique(y)))

        model = RandomForestClassifier()
        model.fit(X, y)
        y_pred = model.predict(X_valid)
        v_error = np.mean(y_pred != y_valid)
        print("Random Forest (sklearn) validation error: %.3f" % v_error)

        model = NaiveBayes(num_classes=4,beta=1)
        model.fit(X, y)
        y_pred = model.predict(X_valid)
        v_error = np.mean(y_pred != y_valid)
        print("Naive Bayes (ours) validation error: %.3f" % v_error)

        model = BernoulliNB()
        model.fit(X, y)
        y_pred = model.predict(X_valid)
        v_error = np.mean(y_pred != y_valid)
        print("Naive Bayes (sklearn) validation error: %.3f" % v_error)

    elif question == '2':
        dataset = load_dataset('vowel.pkl')
        X = dataset['X']
        y = dataset['y']
        X_test = dataset['Xtest']
        y_test = dataset['ytest']
        print("\nn = %d, d = %d\n" % X.shape)

        def evaluate_model(model):
            model.fit(X,y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)
            print("    Training error: %.3f" % tr_error)
            print("    Testing error: %.3f" % te_error)


        print("Our implementations:")
        print("  Decision tree info gain")
        evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))
        print("  Random tree info gain")
        evaluate_model(RandomTree(max_depth=np.inf))
        print("  Random forest info gain")
        #raise NotImplementedError()
        start = time.time()
        evaluate_model(RandomForest(max_depth=np.inf, num_trees=50)) # TODO: implement this
        end = time.time()
        print('running time :{}'.format(end-start))

        print("sklearn implementations")
        print("  Decision tree info gain")
        evaluate_model(DecisionTreeClassifier(criterion="entropy"))
        print("  Random forest info gain")
        evaluate_model(RandomForestClassifier(criterion="entropy"))
        print("  Random forest info gain, more trees")
        start = time.time()
        evaluate_model(RandomForestClassifier(criterion="entropy", n_estimators=50))
        end = time.time()
        print('running time :{}'.format(end-start))


    elif question == '3':
        X = load_dataset('clusterData.pkl')['X']

        model = Kmeans(k=4)
        model.fit(X)
        plot_2dclustering(X, model.predict(X))

        fname = os.path.join("..", "figs", "kmeans_basic.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == '3.1':
        X = load_dataset('clusterData.pkl')['X']

        model = Kmeans(k=4)
        min_error = float('inf')
        fname = os.path.join("..", "figs", "kmeans_min_error.png")

        for i  in range(50):
            error = model.fit(X)
            if error < min_error:
                min_error = error
                plot_2dclustering(X, model.predict(X))
                plt.savefig(fname)

        print('lowest error is {}'.format(min_error))


    elif question == '3.2':
        X = load_dataset('clusterData.pkl')['X']
        fname = os.path.join("..", "figs", "kmeans_3_2.png")

        k_list = list(range(1,11))
        error_list = [None] * 10

        for d in range(1,11):
            model = Kmeans(k=d)
            min_error = float('inf')

            for i in range(50):
                error = model.fit(X)
                if error < min_error:
                    min_error = error

            error_list[d-1] = min_error

        plt.scatter(k_list,error_list)
        plt.title('error vs k')
        plt.savefig(fname)


    elif question == '3.3':
        X = load_dataset('clusterData2.pkl')['X']

        #3.3.1
        '''
        model = Kmeans(k=4)
        min_error = float('inf')
        fname = os.path.join("..", "figs", "kmeans_3_3.png")

        for i  in range(50):
            error = model.fit(X)
            if error < min_error:
                min_error = error
                plot_2dclustering(X, model.predict(X))
                plt.savefig(fname)

        print('lowest error is {}'.format(min_error))

        '''
        #3.3.2
        '''
        fname = os.path.join("..", "figs", "kmeans_3_3_elbow.png")

        k_list = list(range(1,11))
        error_list = [None] * 10

        for d in range(1,11):
            model = Kmeans(k=d)
            min_error = float('inf')

            for i in range(50):
                error = model.fit(X)
                if error < min_error:
                    min_error = error

            error_list[d-1] = min_error


        plt.scatter(range(1,11),error_list)
        plt.title('error vs k')
        plt.savefig(fname)
        '''
        #3.3.3

        model = Kmedians(k =4)
        min_error = float('inf')
        fname = os.path.join("..", "figs", "kmeans_3_3_3.png")

        for i in range(50):
            error = model.fit(X)

            if error < min_error:
                min_error = error
                plot_2dclustering(X, model.predict(X))
                plt.savefig(fname)

        print('lowest error is {}'.format(min_error))

        #3.3.4
        '''
        fname = os.path.join("..", "figs", "kmeans_3_3_4_elbow.png")

        k_list = list(range(1,11))
        error_list = [None] * 10

        for d in range(1,11):
            model = Kmedians(k=d)
            min_error = float('inf')

            for i in range(50):
                error = model.fit(X)
                if error < min_error:
                    min_error = error

            error_list[d-1] = min_error


        plt.scatter(range(1,11),error_list)
        plt.title('error vs k by Kmedians')
        plt.savefig(fname)
        '''


    elif question == '3.4':
        X = load_dataset('clusterData2.pkl')['X']

        model = DBSCAN(eps=200, min_samples=0)
        y = model.fit_predict(X)

        print("Labels (-1 is unassigned):", np.unique(model.labels_))

        plot_2dclustering(X,y)
        fname = os.path.join("..", "figs", "clusterdata_dbscan.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    elif question == '4':
        img = imread(os.path.join("..", "data", "mandrill.jpg"))

        # part 1: implement quantize_image.py
        # part 2: use it on the doge
        for b in [1,2,4,6]:
            quantizer = ImageQuantizer(b)
            q_img = quantizer.quantize(img)
            d_img = quantizer.dequantize(q_img)

            plt.figure()
            plt.imshow(d_img)
            fname = os.path.join("..", "figs", "b_{}_image.png".format(b))
            plt.savefig(fname)
            print("Figure saved as '%s'" % fname)

            plt.figure()
            plt.imshow(quantizer.colours[None] if b/2!=b//2 else np.reshape(quantizer.colours, (2**(b//2),2**(b//2),3)))
            plt.title("Colours learned")
            plt.xticks([])
            plt.yticks([])
            fname = os.path.join("..", "figs", "b_{}_colours.png".format(b))
            plt.savefig(fname)
            print("Figure saved as '%s'" % fname)


    else:
        print("Unknown question: %s" % question)
