import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE


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


def t_sne(y_t, t_num):
    y_t = np.array(y_t)
    y_t_values_flat = y_t.reshape((y_t.shape[0], -1))
    y_t_tsne = TSNE(
        n_components=2, perplexity=1.75).fit_transform(y_t_values_flat)
    # print(y_t_tsne)
    # Plot the t-SNE visualization with color-coded classes
    plt.scatter(y_t_tsne[:, 0], y_t_tsne[:, 1], c=range(0, len(y_t_tsne)),
                cmap='viridis', marker='o', edgecolors='w', s=100)
    plt.title('t-SNE Visualization of 5 Classes at t='+str(t_num))
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    plt.colorbar(label='Class')
    plt.savefig('t_sne_'+str(t_num)+'.png', format='PNG')
    # plt.legend()
    plt.close()
