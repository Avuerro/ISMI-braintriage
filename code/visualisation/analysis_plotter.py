import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn

def plot_roc_curve(fpr, tpr, roc_auc):
    """
        This functions plots the ROC-curve and shows its corresponding AUC

        Params:
            fpr: false positive rate (list)
            tpr: true positive rate (list)
            roc_auc: area under the curve

    """
    plt.figure()
    plt.plot(fpr, tpr, color='deeppink', label='ROC-curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-curve')
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_matrix(targets, predictions):
    """
        This functions plots a confusion matrix based on targets and predictions made by a network

        Params:
            targets: list containing target values [0 or 1]
            predictions: list containing predicted values [0 or 1]

    """
    plt.figure()
    cm = confusion_matrix(targets, predictions)
    labels = ['Abnormal', 'Normal']
    sn.heatmap(cm, annot=True, cmap='RdPu', yticklabels=labels, xticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Target")
    plt.title('Patient Labels Predicted by Network')
    plt.show()
