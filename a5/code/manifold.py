import numpy as np
from numpy.linalg import norm
import utils
from pca import PCA
from findMin import findMin

class MDS:

    def __init__(self, n_components):
        self.k = n_components

    def compress(self, X):
        n = X.shape[0]
        k = self.k

        # Compute Euclidean distances
        D = utils.euclidean_dist_squared(X,X)
        D = np.sqrt(D)

        # Initialize low-dimensional representation with PCA
        pca = PCA(k)
        pca.fit(X)
        Z = pca.compress(X)

        # Solve for the minimizer
        z, f = findMin(self._fun_obj_z, Z.flatten(), 500, D)
        Z = z.reshape(n, k)
        return Z

    def _fun_obj_z(self, z, D):
        n = D.shape[0]
        k = self.k
        Z = z.reshape(n,k)

        f = 0.0
        g = np.zeros((n,k))
        for i in range(n):
            for j in range(i+1,n):
                # Objective Function
                Dz = norm(Z[i]-Z[j])
                s = D[i,j] - Dz
                f = f + (0.5)*(s**2)

                # Gradient
                df = s
                dgi = (Z[i]-Z[j])/Dz
                dgj = (Z[j]-Z[i])/Dz
                g[i] = g[i] - df*dgi
                g[j] = g[j] - df*dgj

        return f, g.flatten()



class ISOMAP(MDS):

    def __init__(self, n_components, n_neighbours):
        self.k = n_components
        self.nn = n_neighbours


    def construct_dist_graph(self, X , D):
        n,d = X.shape
        graph = np.zeros((n,n), dtype = bool)

        for i in range(n):
            distances = X[i,:] - X
            distances = norm(distances , axis = 1)
            rank = np.argsort(distances)

            #rank[0] is it self
            graph[i , rank[1]] = True
            graph[i , rank[2]] = True

        #mutually KNN
        #graph = np. logical_and(graph,np.transpose(graph))

        weights = graph * D
        weights [weights==0] = np.inf

        return weights


    def compress(self, X):
        n = X.shape[0]

        # Compute Euclidean distances
        D = utils.euclidean_dist_squared(X,X)
        D = np.sqrt(D)

        #TODO:
        D = self.construct_dist_graph(X , D)

        # If two points are disconnected (distance is Inf)
        # then set their distance to the maximum
        # distance in the graph, to encourage them to be far apart.
        D[np.isinf(D)] = D[~np.isinf(D)].max()


        # Initialize low-dimensional representation with PCA
        pca = PCA(self.k)
        pca.fit(X)
        Z = pca.compress(X)

        # Solve for the minimizer
        z,f = findMin(self._fun_obj_z, Z.flatten(), 500, D)
        Z = z.reshape(n, self.k)
        return Z
