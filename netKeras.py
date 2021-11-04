import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import network


class netKeras():
    def __init__(self, epochs, eta):
        self.epochs=epochs
        self.eta=eta
    def train(self, Xtrain, Ytrain, Xtest, Ytest):
        model=keras.Sequential([
            keras.layers.Dense(4,input_shape=(12,),activation='sigmoid'),
            keras.layers.Dense(2, input_shape=(4,), activation='sigmoid'),
            keras.layers.Dense(1,input_shape=(2,),activation='sigmoid')
        ])
        opt = keras.optimizers.SGD(learning_rate=self.eta)
        model.compile(optimizer=opt,
                      loss='mean_squared_error',
                      metrics=[ 'accuracy',tf.keras.metrics.BinaryCrossentropy()])

        model.fit(Xtrain, Ytrain, epochs=self.epochs)
        model.evaluate(Xtest,Ytest)




