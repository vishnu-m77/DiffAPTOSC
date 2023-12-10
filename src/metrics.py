import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
import os


def plot_loss(loss_arr, title, savedir, test_loss_array=None):
    plt.plot(loss_arr, label = "Train Loss")
    if test_loss_array!=None:
        plt.plot(test_loss_array, label = "Test Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
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
    target = target.detach().numpy()
    pred_np = pred.detach().numpy()
    F1 = f1_score(target, pred_np, average='macro')
    return F1


def t_sne(targets, y_t, t_num):
    y_t = np.array(y_t)
    # print(targets)
    y_t_values_flat = y_t.reshape((y_t.shape[0], -1))
    y_t_tsne = TSNE(
        n_components=2, perplexity=1.8).fit_transform(y_t_values_flat)
    # print(y_t_tsne)
    # Plot the t-SNE visualization with color-coded classes
    plt.scatter(y_t_tsne[:, 0], y_t_tsne[:, 1], c=targets,
                cmap='viridis', marker='o', edgecolors='w', s=100)
    plt.title('t-SNE Visualization of 5 Classes at t='+str(t_num))
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    plt.colorbar(label='Class')
    plt.savefig('plots/t_sne_'+str(t_num)+'.png', format='PNG')
    # plt.legend()
    plt.close()
    return y_t_tsne


if __name__ == "__main__":
    """
    TSNE with class labels
    """
    # The y_t data
    y_t = np.array([[0, 2, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0.5, 0],
                    [0, 0, 0, 0, 1.2],
                    [0, 0, 0, 0, -1],
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 10, 0, 0],
                    [0, 20, 0, 0, 0]])

    # The corresponding labels for
    y_class = [0, 1, 2, 3, 4, 2, 2, 3, 3]
    if len(y_class) != len(y_t):
        raise KeyError("Size mismatch between data and labels")
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    y_t_tsne = tsne.fit_transform(y_t)

    # Plot the t-SNE results
    plt.figure(figsize=(8, 6))
    legend_labels = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']
    color = ['blue', 'red', 'orange', 'purple', 'green']
    # Scatter plot with class labels
    for i in range(len(y_class)):
        indices = y_class[i]
        plt.scatter(y_t_tsne[i, 0], y_t_tsne[i, 1],
                    label=legend_labels[indices], c=color[indices])

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Show the plot
    plt.title('t-SNE Visualization of Data with Class Labels')
    plt.show()
