
import utils
import numpy as np
from random_tree import RandomTree


class RandomForest:

    def __init__(self, num_trees, max_depth):
        self.num_trees = num_trees
        self.max_depth = max_depth

    def fit(self, X, y):

        forest = []

        for n in range(self.num_trees):

            model = RandomTree( max_depth = self.max_depth)
            model.fit(X,y)
            forest.append(model)

        self.forest = forest

    def predict (self, X):

        y_pred_list = []
        y_pred = []

        N,D = X.shape

        for n in range(self.num_trees):

            y_pred_list.append((self.forest[n]).predict(X))


        y_pred = [None] * N
        l = []
        for n in range(N):
            for t in range(self.num_trees):
                l .append( y_pred_list[t][n] )
            y_pred[n] = utils.mode(np.array(l))
            l = []

        return y_pred



