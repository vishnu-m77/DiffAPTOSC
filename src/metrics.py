import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score


def plot_loss(loss_arr, title, xlabel, ylabel, savedir):
    plt.plot(loss_arr)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(savedir+'.png', format='PNG')
    plt.close()


def accuracy_torch(tensor_one, tensor_two):
    """
    Takes in 2 tensors and computes accuracy
    """
    correct = 0
    if tensor_one.size(0) != tensor_two.size(0):
        raise KeyError("Tensor dimension mismatch")
    for idx in range(tensor_one.size(0)):
        if tensor_one[idx] == tensor_two[idx]:
            correct = correct + 1
    return correct/tensor_one.size(0)


def compute_f1_score(target, pred):
    target = target.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()
    # pred_np = np.argmax(pred_np, axis=1)
    F1 = f1_score(target, pred_np, average='macro')
    return F1
