import numpy as np

def get_MSE(pred, real):
    return np.mean(np.power(real - pred, 2))

def get_RMSE(pred, real):
    return np.sqrt(np.mean(np.power(real - pred, 2)))

def get_MAE(pred, real):
    return np.mean(np.abs(real - pred))
    
def get_NRMSE(pred, real):
    return np.sqrt(np.mean(np.power(real - pred, 2))) / np.mean(np.abs(real))