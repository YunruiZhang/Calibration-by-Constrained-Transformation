import numpy as np
from scipy.optimize import minimize 

from sklearn.metrics import log_loss, mean_squared_error
from os.path import join
import sklearn.metrics as metrics



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

class MCCT():
    
    def __init__(self, maxiter=100, solver="SLSQP", topk=10, loss = "ce", bounds = False, filter = False, tol = 1e-12):
        """
        Initialize class
        
        Params:
            maxiter (int): maximum iterations done by optimizer, however 8 iterations have been maximum.
            solver (str): optimization algorithm used by scipy.optimize.minimize
            topk (int): number of classes to be considered for monotonic vector scaling
        """
        self.maxiter = maxiter
        self.solver = solver
        self.topk = topk
        self.loss = loss
        self.bounds = bounds
        self.filter = filter
        self.tol = tol
    
    def _loss_fun(self, x, probs, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        scaled_probs = self.predict_train(probs, temp=x[:self.topk], bias=x[self.topk:])
        if self.loss == "ce":
            #loss = log_loss(y_true=true, y_pred=scaled_probs)
            loss = -np.sum(true*np.log(scaled_probs))/probs.shape[0]
        elif self.loss == "mse":
            loss = mean_squared_error(y_true=true, y_pred=scaled_probs)    

        return loss
    
        # probs is the logits
    def _jac_fun(self, x, probs, true):
        """
        Compute the Jacobian of the loss with respect to the parameters (T and B).
        """
        temp = x[:self.topk]  # Temperature parameters
        bias = x[self.topk:]  # Bias parameters
        
        # Forward pass to compute scaled probabilities
        scaled_probs = self.predict_train(probs, temp, bias)
        
        # Gradients of the loss
        grad_T = np.zeros(self.topk)
        grad_B = np.zeros(self.topk)
        
        # Loop over samples to compute gradients
        for i in range(probs.shape[0]):
            p = scaled_probs[i]  # Softmax probabilities for sample i
            y = true[i]  # True one-hot encoding for sample i
            logits = probs[i]
            
            # Derivative of loss with respect to logits (softmax gradient)
            grad_logits = p - y  # Shape: [topk]
            
            # Gradients w.r.t temperature (T)
            grad_T += (logits * grad_logits)
            
            # Gradients w.r.t bias (B)
            grad_B += grad_logits  # Since d(logits + B)/dB = 1

        grad_T /= probs.shape[0]
        grad_B /= probs.shape[0]
        
        # Combine gradients into one vector
        jacobian = np.concatenate((grad_T, grad_B), axis=0)
        return jacobian
    
    def ascending_constraints(self, params, topk):
        # Extract temperature and bias from params
        t = params[:topk]
        b = params[topk:]
        
        # Constraints: t[i] > t[i-1] and b[i] > b[i-1]
        temp_constraints = np.diff(t)  # Differences for temperature
        bias_constraints = np.diff(b)  # Differences for bias
        
        # Return the concatenated array of differences, which should be all positive
        return np.concatenate((temp_constraints, bias_constraints))
    
    # Find the temperature and bias
    def fit(self, logits, true):
        """
        Trains the model and finds optimal temperature and bias
        
        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: true labels in one hot encoding need to sort this too.
            
        Returns:
            the results of optimizer after minimizing is finished.
        """

        # maybe consider normalizing the logits before applying the scaling?
        self.nclass = logits.shape[1]
        sorted_logits = np.sort(logits, axis=1)

        logit_sort_index = np.argsort(logits, axis=1)
        true_sorted = true.copy()
        for a in range(logit_sort_index.shape[0]):
            true_sorted[a] = true_sorted[a][logit_sort_index[a]]

        # only consider topk classes check if the labels are in the tok k classes
        # do we need to take them out? The ones that the labels are not in topk classes? Should not make any difference should it?
        # for now we do not take them out justkeep them might be a problem for imagenet
        sorted_logits = sorted_logits[:, -self.topk:]
        true_sorted = true_sorted[:, -self.topk:]

        if self.filter == True:
            # only consider topk classes check if the labels are in the tok k classes
            indicator = true_sorted.sum(axis=1)
            index_of_match = np.where(indicator == 1)[0]
            sorted_logits = sorted_logits[index_of_match, :]
            true_sorted = true_sorted[index_of_match, :]

        temp_init = np.linspace(0, 1, num=self.topk)
        # temp_init[-1] = 1
        bias_init = np.zeros(self.topk)
        # bias_init = np.linspace(0, 1, num=self.topk)

        
        initial_guess = np.concatenate((temp_init, bias_init), axis=0)
        constraints = {'type': 'ineq', 'fun': self.ascending_constraints, 'args': (self.topk,)}

        if self.bounds == True:
            bounds_weight = [(0, 1) for i in range(self.topk)]
            bounds_bias = [(-1, 1) for i in range(self.topk)]
            bounds = bounds_weight + bounds_bias
            opt = minimize(self._loss_fun, x0=initial_guess, args=(sorted_logits, true_sorted), 
                       options={'maxiter':self.maxiter, 'disp':True}, method=self.solver, constraints=constraints, bounds=bounds, tol=self.tol, jac=self._jac_fun)
        else:
            bounds_weight = [(0, None) for i in range(self.topk)]
            bounds_bias = [(None, None) for i in range(self.topk)]
            bounds = bounds_weight + bounds_bias
            opt = minimize(self._loss_fun, x0=initial_guess, args=(sorted_logits, true_sorted), 
                       options={'maxiter':self.maxiter, 'disp':True}, method=self.solver, constraints=constraints, bounds=bounds, tol=self.tol, jac=self._jac_fun)
        
        self.temp = opt.x[:self.topk]  # Update temperature
        self.bias = opt.x[self.topk:]  # Update bias
        
        return opt
        
    def predict_train(self, logits, temp, bias):
        """
        Scales logits based on the temperature and adds bias, returning calibrated probabilities
        
        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            temp: if not set, uses temperature found by the model or previously set.
            bias: if not set, uses bias found by the model or previously set.
            
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        
        # Scale logits by temperature and add bias
        scaled_logits = (logits * temp) + bias
        return softmax(scaled_logits)
    
    def predict(self, logits):
        # sort the logits and keep the index

        logits_sorted = np.sort(logits, axis=1)
        logit_sort_index = np.argsort(logits, axis=1)

        logits_sorted_k = logits_sorted[:, -self.topk:]
        # logit_sort_index_k = logit_sort_index[:, -self.topk:]

        scaled_logits = (logits_sorted_k * self.temp) + self.bias
        # return the logits to the original order
        scaled_logits_original = np.zeros_like(logits)

        min_temp = self.temp[0]
        min_bias = self.bias[0]
        for a in range(logit_sort_index.shape[0]):
            for i in range(logit_sort_index.shape[1]):
                # check if the index is in the topk classes
                # keep putting zeros untile reach topk
                if i < self.nclass - self.topk:
                    # keep zero or put the original value?
                    scaled_logits_original[a][logit_sort_index[a][i]] = logits_sorted[a][i] * min_temp + min_bias
                else:
                    scaled_logits_original[a][logit_sort_index[a][i]] = scaled_logits[a][self.topk - (self.nclass - i)]

        return softmax(scaled_logits_original)


class MCCT_I():
    
    def __init__(self, maxiter=100, solver="SLSQP", topk=10, loss = "ce", bounds = False, filter = False, tol = 1e-12):
        """
        Initialize class
        
        Params:
            maxiter (int): maximum iterations done by optimizer, however 8 iterations have been maximum.
            solver (str): optimization algorithm used by scipy.optimize.minimize
            topk (int): number of classes to be considered for monotonic vector scaling
        """
        self.maxiter = maxiter
        self.solver = solver
        self.topk = topk
        self.loss = loss
        self.bounds = bounds
        self.filter = filter
        self.tol = tol
    
    def _loss_fun(self, x, probs, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        scaled_probs = self.predict_train(probs, temp=x[:self.topk], bias=x[self.topk:])
        if self.loss == "ce":
            #loss = log_loss(y_true=true, y_pred=scaled_probs)
            loss = -np.sum(true*np.log(scaled_probs))/probs.shape[0]
        elif self.loss == "mse":
            loss = mean_squared_error(y_true=true, y_pred=scaled_probs)    

        return loss
    
    def _jac_fun(self, x, probs, true):
        """
        Compute the Jacobian of the loss with respect to the parameters (T and B).
        """
        temp = x[:self.topk]  # Temperature parameters
        bias = x[self.topk:]  # Bias parameters
        
        # Forward pass to compute scaled probabilities
        scaled_probs = self.predict_train(probs, temp, bias)
        
        # Gradients of the loss
        grad_T = np.zeros(self.topk)
        grad_B = np.zeros(self.topk)
        
        # Loop over samples to compute gradients
        for i in range(probs.shape[0]):
            p = scaled_probs[i]  # Softmax probabilities for sample i
            y = true[i]  # True one-hot encoding for sample i
            logits = probs[i]
            
            # Derivative of loss with respect to logits (softmax gradient)
            grad_logits = p - y  # Shape: [topk]
            
            # Gradients w.r.t temperature (T)
            grad_T += -1 / (temp ** 2) * (logits * grad_logits)
            
            # Gradients w.r.t bias (B)
            grad_B += grad_logits  # Since d(logits + B)/dB = 1

        grad_T /= probs.shape[0]
        grad_B /= probs.shape[0]
        
        # Combine gradients into one vector
        jacobian = np.concatenate((grad_T, grad_B), axis=0)
        return jacobian
    
    
    def ascending_constraints(self, params, topk):
        # Extract temperature and bias from params
        t = params[:topk]
        b = params[topk:]
        
        t = t[::-1]
        # Constraints: t[i] > t[i-1] and b[i] > b[i-1]
        temp_constraints = np.diff(t)  # Differences for temperature
        bias_constraints = np.diff(b)  # Differences for bias
        
        # Return the concatenated array of differences, which should be all positive
        return np.concatenate((temp_constraints, bias_constraints))
    
    # Find the temperature and bias
    def fit(self, logits, true):
        """
        Trains the model and finds optimal temperature and bias
        
        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: true labels in one hot encoding need to sort this too.
            
        Returns:
            the results of optimizer after minimizing is finished.
        """

        # maybe consider normalizing the logits before applying the scaling?
        self.nclass = logits.shape[1]
        sorted_logits = np.sort(logits, axis=1)

        logit_sort_index = np.argsort(logits, axis=1)
        true_sorted = true.copy()
        for a in range(logit_sort_index.shape[0]):
            true_sorted[a] = true_sorted[a][logit_sort_index[a]]

        # only consider topk classes check if the labels are in the tok k classes
        # do we need to take them out? The ones that the labels are not in topk classes? Should not make any difference should it?
        # for now we do not take them out justkeep them might be a problem for imagenet
        sorted_logits = sorted_logits[:, -self.topk:]
        true_sorted = true_sorted[:, -self.topk:]

        if self.filter == True:
            # only consider topk classes check if the labels are in the tok k classes
            indicator = true_sorted.sum(axis=1)
            index_of_match = np.where(indicator == 1)[0]
            sorted_logits = sorted_logits[index_of_match, :]
            true_sorted = true_sorted[index_of_match, :]

        temp_init = np.ones(self.topk)
        # temp_init[-1] = 1
        bias_init = np.zeros(self.topk)
        # bias_init = np.linspace(0, 1, num=self.topk)

        
        initial_guess = np.concatenate((temp_init, bias_init), axis=0)
        constraints = {'type': 'ineq', 'fun': self.ascending_constraints, 'args': (self.topk,)}

        if self.bounds == True:
            bounds_weight = [(0, 4) for i in range(self.topk)]
            bounds_bias = [(-1, 1) for i in range(self.topk)]
            bounds = bounds_weight + bounds_bias
            opt = minimize(self._loss_fun, x0=initial_guess, args=(sorted_logits, true_sorted), 
                       options={'maxiter':self.maxiter, 'disp':True}, method=self.solver, constraints=constraints, bounds=bounds, tol=self.tol, jac=self._jac_fun)
        else:
            bounds_weight = [(0, None) for i in range(self.topk)]
            bounds_bias = [(None, None) for i in range(self.topk)]
            bounds = bounds_weight + bounds_bias
            opt = minimize(self._loss_fun, x0=initial_guess, args=(sorted_logits, true_sorted), 
                       options={'maxiter':self.maxiter, 'disp':True}, method=self.solver, constraints=constraints, bounds=bounds, tol=self.tol, jac=self._jac_fun)
        
        self.temp = opt.x[:self.topk]  # Update temperature
        self.bias = opt.x[self.topk:]  # Update bias
        
        return opt
        
    def predict_train(self, logits, temp, bias):
        """
        Scales logits based on the temperature and adds bias, returning calibrated probabilities
        
        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            temp: if not set, uses temperature found by the model or previously set.
            bias: if not set, uses bias found by the model or previously set.
            
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        
        # Scale logits by temperature and add bias
        scaled_logits = (logits / temp) + bias
        return softmax(scaled_logits)
    
    def predict(self, logits):
        # sort the logits and keep the index

        logits_sorted = np.sort(logits, axis=1)
        logit_sort_index = np.argsort(logits, axis=1)

        logits_sorted_k = logits_sorted[:, -self.topk:]
        # logit_sort_index_k = logit_sort_index[:, -self.topk:]

        scaled_logits = (logits_sorted_k / self.temp) + self.bias
        # return the logits to the original order
        scaled_logits_original = np.zeros_like(logits)

        min_temp = self.temp[0]
        min_bias = self.bias[0]
        for a in range(logit_sort_index.shape[0]):
            for i in range(logit_sort_index.shape[1]):
                # check if the index is in the topk classes
                # keep putting zeros untile reach topk
                if i < self.nclass - self.topk:
                    # keep zero or put the original value?
                    scaled_logits_original[a][logit_sort_index[a][i]] = logits_sorted[a][i] / min_temp + min_bias
                else:
                    scaled_logits_original[a][logit_sort_index[a][i]] = scaled_logits[a][self.topk - (self.nclass - i)]

        return softmax(scaled_logits_original)
