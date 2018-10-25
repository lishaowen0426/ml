import numpy as np

class Kmedians:

    def __init__(self, k):
        self.k = k


    def fit(self, X):

        N,D = X.shape
        y = np.ones(N)

        error = None

        medians = np.zeros((self.k,D))

        for kk in range(self.k):
            i = np.random.randint(N)
            medians[kk] = X[i]

        while True:

            y_old = y

            for n in range(N):
                dist = np.zeros(self.k)
                for k in range(self.k):
                    l = medians[k] - X[n]
                    dist[k] = np.sum( [abs(x) for x in l] )

                y[n] = np.argmin(dist)

            #update medians
            for kk in range(self.k):
                medians[kk] = np.median(X[y==kk],axis = 0)

            changes = np.sum(y != y_old)

            self.medians = medians

            if changes == 0:
                error = self.error(X)
                break

        self.medians = medians
        return error

    def predict(self, X):

        medians = self.medians
        N,D = X.shape

        y = np.zeros(N)

        for n in range(N):
            dist = np.zeros(self.k)

            for k in range(self.k):
                l = medians[k] - X[n]
                dist[k] = np.sum([abs(x) for x in l])

            y[n] = np.argmin(dist)

        return y


    def error (self, X):

        N,D = X.shape
        medians = self.medians
        sum = 0

        for n in range(N):
            dist = np.zeros(self.k)

            for k in range(self.k):
                l = medians[k]-X[n]
                dist[k] = np.sum([abs(x) for x in l])

            sum += np.min(dist)

        return sum


