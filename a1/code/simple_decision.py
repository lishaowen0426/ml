import numpy as np
from decision_stump import DecisionStump

def predict (self,X):
    M, D = X.shape
    y = np.zeros(M)

    splitVariable = self.splitModel.splitVariable
    splitValue = self.spltModel.splitValue
    splitSat = self.splitModel.splitSat


    if splitVariable is None:
        y=splitSat * np.ones(M)


    elif self.subModel1 is None:
        return self.splitModel.predict(X)

    else:

        j_root = splitVariable
        value_root = splitValue

        splitIndex1 = X[:,j_root] > value_root
        splitIndex0 = X[:,j_root] <= value_root

        j_1=self.subModel1.splitVariable
        value_1=self.subModel1.splitValue


        j_0=self.subModel0.splitVariable
        value_0=self.subModel0.splitValue

        y[ X[splitIndex1, j_1] > value_1 ] = self.splitModel1.splitSat
        y[ X[splitIndex1,j_1] <= value_1 ] = self.splitModel1.splitNot

        y[ X[splitIndex0, j_0] > value_0 ] = self.splitModel0.splitSat
        y[ X[splitIndex0, j_0] <= value_0 ] = self.splitModel0.splitNot


    return y
