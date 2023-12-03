import numpy as np
from matplotlib import pyplot as plt
# import sklearn.metrics as metrics
# #from imblearn.metrics import sensitivity_score, specificity_score
# import pdb
# # from sklearn.metrics.ranking import roc_auc_score
# from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
# from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def plot_loss(loss_arr, title, xlabel, ylabel, savedir):
    plt.plot(loss_arr)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(savedir+'.png', format='PNG')
    plt.close()
