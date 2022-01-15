import matplotlib as mpl
import matplotlib.pyplot as plt


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