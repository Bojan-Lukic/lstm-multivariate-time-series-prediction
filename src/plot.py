import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from data import create_time_steps


def multi_step_plot(history, true_future, prediction, title, STEP):
    calibri = {'fontname':'Calibri'}
    plt.figure(figsize=(12, 6))
    
    num_in = create_time_steps(len(history))
    num_out = len(true_future)
    
    ##########################
    difference = []
    zip_object = zip(true_future, prediction)
    for true_future_i, prediction_i in zip_object:
        difference.append(true_future_i - prediction_i)

    difference_abs = [abs(ele) for ele in difference] 
    
    me_diff = max(difference_abs) #me = "maximum error" = "Maximaler Einzelfehler" ???
    sd_diff = np.sqrt(np.mean(np.power(difference, 2))) #sd = "standard deviation" = "Standardabweichung"
    rmse = sqrt(mean_squared_error(true_future, prediction))
    mae = mean_absolute_error(true_future, prediction) #mae = "mean absolute error"
    ##########################
    
    plt.plot(num_in, np.array(history[:, 1]), label='History')
    plt.plot((np.arange(num_out) + 1)/STEP, np.array(true_future), 'bo', label='True Future')
    
    if prediction.any():
        plt.plot(np.arange(num_out) + 1/STEP, np.array(prediction), 'ro', label='Predicted Future')
        
    plt.legend(loc='upper left', fontsize=14)
    plt.xlabel('Time steps (h)', **calibri)
    plt.ylabel('Temperature (°C)', **calibri)
    plt.grid()
    plt.title(title, **calibri)
    mpl.rcParams.update({'font.size': 20})
    plt.show()
    print('RMSE: %.3f' % rmse)
    print('MAE: %.3f' % mae)
    print('ME: %.3f' % me_diff)
    print('SD: %.3f' % sd_diff)
    
    
def plot_train_history(history, title):
    calibri = {'fontname':'Calibri'}
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(loss))

    plt.figure(figsize=(12, 8))

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title, **calibri)
    plt.xlabel('Epochs', **calibri)
    plt.xticks([-1, 4, 9, 14, 19, 24, 29], [0, 5, 10, 15, 20, 25, 30])
    plt.ylabel('MSE loss', **calibri)
    plt.grid()
    plt.legend(fontsize=16)
    mpl.rcParams.update({'font.size': 20})

    plt.show()