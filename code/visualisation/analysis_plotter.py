import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import numpy as np

def plot_roc_curve(fpr, tpr, roc_auc, color='deeppink'):
    """
        This functions plots the ROC-curve and shows its corresponding AUC

        Params:
            fpr: false positive rate (list)
            tpr: true positive rate (list)
            roc_auc: area under the curve

    """
    plt.figure()
    plt.plot(fpr, tpr, color=color, label='ROC-curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_matrix(targets, predictions, title, show_cbar=True, show_annot=True, cmap="Blues"):
    """
        This functions plots a confusion matrix based on targets and predictions made by a network

        Params:
            targets: list containing target values [0 or 1]
            predictions: list containing predicted values [0 or 1]
            title: title to put at the top of the confusion matrix
            xticklabels/yticklabels: labels to use on the x/y axis

    """
    plt.figure()
    cm = confusion_matrix(targets, predictions)
    sn.set(font_scale=1.4)
    labels = ['Normal', 'Abnormal']
    sn.heatmap(cm, annot=show_annot, cmap=cmap, yticklabels=False, xticklabels=labels, fmt="d", cbar=show_cbar)
    plt.yticks(np.arange(len(labels))+0.5,labels)
    plt.xlabel("Predicted")
    plt.ylabel("Target")
    plt.title(title)
    plt.show()
