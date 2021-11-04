import random

import numpy as np


class Network():
    # Sizes define la estructura de la red y el numero de neuronas por capa e.g Network [2,3,1]
    # Eta, tasa aprendizaje
    # Epochs, iteraciones del training dataset completo
    # Size batch
    def __init__(self, sizes,eta,epochs, sizeBatch):
        self.num_layer=len(sizes)
        self.sizes=sizes
        self.bias=[np.random.randn(y,1) for y in sizes[1:]] # Crea los bias para todas las capas que no son input
        self.weights= [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])] # weights[0] conexiones il-hl, weights[1]_hl-ol
        self.eta=eta
        self.epochs=epochs
        self.sizeBatch=sizeBatch

    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))

    def derivSigmoid(self, sig):
        # dA/dZ
        return sig *(1-sig)

    def cuadraticError(self,output, target):
        return 0.5 * np.square(target-output)
    def derivError(self, output, y):
        # dC/dA Error cuadrático
        return  output-y

    def CrossEntropy(self, activationOutput):
        return -np.log(activationOutput)

    def feedForward(self,a):
        for b,w in zip(self.bias, self.weights):
            a=self.sigmoid(np.dot(w,a)+b)
        return a


    def SGD(self, trainData):
        # Descenso del gradiente estocástico
        # traindata es una tupla con pares de la entrada y valores deseados
        for j in range(self.epochs):
            miniBatches=[]
            for k in range(0, self.sizeBatch):
                row=random.randint(0,len(trainData[0])-1)
                miniBatches.append((trainData[0][row],trainData[1][row]))
            #print(miniBatches)
            for miniBatch in miniBatches:
                #print("Mini 1", miniBatch)
                self.updateBatch(miniBatch)
                print("Error", self.cuadraticError(self.activation,miniBatch[1]))

            print("Epoch {} complete".format(j))

    def updateBatch(self, miniBatch):
        # Actualiza los pesos y bias acorde a los datos del batch haciendo uso del backpropagation
        nablaB= [np.zeros(b.shape) for b in self.bias] # nablaB y nablaW bias y pesos de cada capa
        nablaW= [np.zeros(w.shape) for w in self.weights]

        # Devuelve el gradiente para cada muestra del batch (dC/dB y dC/dW)
        x=miniBatch[0]
        y=miniBatch[1]
        deltaNablaB, deltanablaW= self.backprop(x,y)
        #print("Deltas",deltanablaW,deltaNablaB)
        nablaB= [nb+dnb for nb, dnb in zip(nablaB, deltaNablaB)]
        nablaW= [nw+dnw for nw, dnw in zip(nablaW, deltanablaW)]

        # Actualiza los pesos teniendo en cuenta que se debe dividir delta entre el tamaño del batch
        self.weights=[w-(self.eta/self.sizeBatch)*nw for w,nw in zip(self.weights,nablaW)]
        self.bias=[b-(self.eta/self.sizeBatch)*nb for b,nb in zip(self.bias,nablaB)]



    def backprop(self,x,y):
        nablaB = [np.zeros(b.shape) for b in self.bias]
        nablaW = [np.zeros(w.shape) for w in self.weights]

        # Feedforward
        self.activation=x # Activacion será la entrada
        #print("Entrada",x)
        self.activations=[x] # Activaciones de cada capa
        zs=[] # Z de cada capa
        for b,w in zip(self.bias, self.weights):
            # print("bias",b)
            # print("pesos",w[0])
            # print("activ", self.activation)
            # print("Suma de pesos", np.dot(w,activation.transpose()))
            z=np.dot(w,self.activation.transpose())+b # Suma ponderada +b[0]
            zs.append(z)
            self.activation=self.sigmoid(z) # sig(z)= activacion
            self.activations.append(self.activation)

        # Retropropagación
        # Delta de la neurona de salida será dC/DA* dA/dZ
        delta=self.derivError(self.activations[-1],y) * self.derivSigmoid(zs[-1])
        nablaB[-1]=delta # dZ/dB = 1
        nablaW[-1]= np.dot(delta, self.activations[-2].T)
        # dZ/dW = activacion de la capa anterior

        for l in range(2, self.num_layer):
            z=zs[-l]
            sigDer = self.derivSigmoid(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sigDer
            nablaB[-l]=delta
            # print("D",delta[0])
            # print(self.activations[-l-1])
            #print("suma nablaw",np.sum(delta[0]*self.activations[-l-1]))
            nablaW[-1] = np.sum(delta[0]*self.activations[-l-1])
            return (nablaB,nablaW)