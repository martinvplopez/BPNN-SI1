import network
import numpy as np
X= np.array([[1,2,3],[4.5,3,1],[4,2,2.2],[3.5,4,6],[3,4,0.7],[2,2,2],[9,6,3.5],[6,1.5,1.8]])
Y= np.array([1,0,1,1,1,1,0,1])
training=list((X,Y))

BPNN=network.Network([3,1,1], 0.01, 500, 4)
BPNN.SGD(training)