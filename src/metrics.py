import os
import logging
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE

def plot_confusion(expected, predicted, mode = True):
    num_classes = np.unique(expected)
    labels = range(np.max(num_classes))
    
    cm = confusion_matrix(expected, predicted)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = labels)
    disp.plot()
    os.makedirs('plots', exist_ok=True)
    if mode:
        plt.title('Confusion Matrix for DCG')
        plt.savefig('plots/dcg_confusion'+'.png', format='PNG')
    else:
        plt.title('Confusion Matrix for Diffusion')
        plt.savefig('plots/diff_confusion'+'.png', format='PNG')
    plt.close()

def plot_loss(loss_arr, val_loss_array=None, mode = "dcg"):
    
    if mode != "dcg" and mode != "diffusion":
        logging.error('Check value of the mode for plotting. Loss plots not generated.')
        return
    if len(loss_arr) >= 50:
        loss_arr = loss_arr[30:]
        val_loss_array = val_loss_array[30:]
    
    fig, ax = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    ax[0].set_xlabel('Iterations')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Train Loss')
    ax[0].plot(loss_arr, label = "Train Loss")
    
    ax[1].set_xlabel('Iterations')
    ax[1].set_ylabel('Loss')
    ax[1].set_title('Validation Loss')
    ax[1].plot(val_loss_array, label = "Val Loss")
    
    os.makedirs('plots', exist_ok=True)
    if mode == "dcg":
        plt.savefig('plots/dcg_loss'+'.png', format='PNG')
    else:
        plt.savefig('plots/diffusion_loss'+'.png', format='PNG')
    plt.close()


def accuracy_torch(targets, predicted):
    """
    Takes in 2 tensors and computes accuracy
    """
    correct = 0
    if targets.size(0) != predicted.size(0):
        raise KeyError("Tensor dimension mismatch")
    for idx in range(predicted.size(0)):
        if targets[idx] == predicted[idx]:
            correct = correct + 1
    return correct/predicted.size(0)


def compute_f1_score(target, pred):
    target = target.detach().numpy()
    pred_np = pred.detach().numpy()
    F1 = f1_score(target, pred_np, average='macro')
    return F1


def t_sne(targets, y_t, t_num):
    y_t = np.array(y_t)
    y_t_values_flat = y_t.reshape((y_t.shape[0], -1))
    y_t_tsne = TSNE(
        n_components=2, perplexity=1.8).fit_transform(y_t_values_flat)
    # Plot the t-SNE visualization with color-coded classes
    plt.scatter(y_t_tsne[:, 0], y_t_tsne[:, 1], c=targets,
                cmap='viridis', marker='o', edgecolors='w', s=100)
    plt.title('t-SNE Visualization of 5 Classes at t='+str(t_num))
    plt.colorbar(label='Class')
    plt.savefig('plots/t_sne_'+str(t_num)+'.png', format='PNG')
    plt.close()
    return y_t_tsne

def call_metrics(params, targets, dcg_output, diffusion_output, y):
    
    t1, t2, t3 = params["t_sne"]["t1"], params["t_sne"]["t2"], params["t_sne"]["t3"]
    y_outs = y[0]
    y_outs_1 = y[1]
    y_outs_2 = y[2]
    y_outs_3 = y[3]
    dcg_accuracy = accuracy_torch(targets, dcg_output)
    diffusion_accuracy = accuracy_torch(targets, diffusion_output)
    dcg_diffusion_accuracy = accuracy_torch(dcg_output, diffusion_output)
    plot_confusion(targets, dcg_output)
    plot_confusion(targets, diffusion_output, 0)
    f1_score = compute_f1_score(targets, diffusion_output)
    tsne_0 = t_sne(targets, y_outs, t_num='0')
    tsne_1 = t_sne(targets, y_outs_1, t_num=t1)
    tsne_2 = t_sne(targets, y_outs_2, t_num=t2)
    tsne_3 = t_sne(targets, y_outs_3, t_num=t3)
    logging.info("DCG accuracy {}".format(dcg_accuracy))
    logging.info("Diffusion model accuracy {}".format(diffusion_accuracy))
    logging.info("Diffusion-DCG accuracy {}".format(dcg_diffusion_accuracy))
    logging.info("F1 Score {}".format(f1_score))
    
    # Creates a report file
    report_file = 'report.txt'
    if os.path.exists(report_file):
        os.remove(report_file)
    
    f = open(report_file, 'w')
    f.write("Accuracy:\n")
    f.write("DCG model accuracy: \t {}\n".format(dcg_accuracy))
    f.write("Diffusion model accuracy: \t {}\n".format(diffusion_accuracy))
    f.write("Diffusion-DCG accuracy: \t {}\n".format(dcg_diffusion_accuracy))
    f.write("F1 Score: \t {}\n".format(f1_score))
    f.write("T-SNE at time '0' {}".format(tsne_0))
    f.write("T-SNE at time '{}' {}".format(t1, tsne_1))
    f.write("T-SNE at time '{}' {}".format(t2, tsne_2))
    f.write("T-SNE at time '{}' {}".format(t3, tsne_3))
    f.close()

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
