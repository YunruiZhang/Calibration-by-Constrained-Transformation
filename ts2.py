# another implementation of temp scaling that only requires the logits
import numpy as np
from scipy.optimize import minimize 

import pandas as pd
import time
from sklearn.metrics import log_loss, mean_squared_error
from os.path import join
import sklearn.metrics as metrics

# there is a bug in this 
# def softmax(x):
#     """
#     Compute softmax values for each sets of scores in x.
    
#     Parameters:
#         x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
#     Returns:
#         x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
#     """
#     e_x = np.exp(x - np.max(x))  # Subtract max so biggest is 0 to avoid numerical instability
    
#     # Axis 0 if only one dimensional array
#     axis = 0 if len(e_x.shape) == 1 else 1
    
#     return e_x / e_x.sum(axis=axis, keepdims=1)
    

def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    
    Parameters:
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    Returns:
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """
    e_x = np.exp(x - np.max(x,axis = 1).reshape(x.shape[0],1))  # Subtract max so biggest is 0 to avoid numerical instability
    
    # Axis 0 if only one dimensional array
    axis = 0 if len(e_x.shape) == 1 else 1
    
    return e_x / e_x.sum(axis=axis, keepdims=1)

class TemperatureScaling():
    
    def __init__(self, temp = 1, maxiter = 50, solver = "BFGS"):
        """
        Initialize class
        
        Params:
            temp (float): starting temperature, default 1
            maxiter (int): maximum iterations done by optimizer, however 8 iterations have been maximum.
        """
        self.temp = temp
        self.maxiter = maxiter
        self.solver = solver
    
    def _loss_fun(self, x, probs, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        scaled_probs = self.predict(probs, x)    
        loss = log_loss(y_true=true, y_pred=scaled_probs)
        return loss
    
    # Find the temperature
    def fit(self, logits, true):
        """
        Trains the model and finds optimal temperature
        
        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: true labels.
            
        Returns:
            the results of optimizer after minimizing is finished.
        """
        
        # true = true.flatten() # Flatten y_val
        opt = minimize(self._loss_fun, x0 = 1, args=(logits, true), options={'maxiter':self.maxiter}, method = self.solver)
        self.temp = opt.x[0]
        
        return opt
        
    def predict(self, logits, temp = None):
        """
        Scales logits based on the temperature and returns calibrated probabilities
        
        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            temp: if not set use temperatures find by model or previously set.
            
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        
        if not temp:
            # print("Cal. temperature is", self.temp)
            return softmax(logits/self.temp)
        else:
            return softmax(logits/temp)




