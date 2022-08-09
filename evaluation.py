from sklearn.metrics import accuracy_score, confusion_matrix 
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import f1_score, roc_auc_score

def get_clf_eval(y_test, pred, pred_proba):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba)

    print('Confusion Matrix:\n', confusion, sep='')
    print('Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, f1_score: {:.4f}, AUC: {:.4f}'.format(accuracy, precision, recall, f1, roc_auc))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import precision_recall_curve
import numpy as np

# %matplotlib inline

def precision_recall_curve_plot(Y_test, pred_proba_class1):
    precisions, recalls, thresholds = precision_recall_curve(Y_test, pred_proba_class1)
    
    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[:threshold_boundary], linestyle='--', label='precision')
    plt.plot(thresholds, recalls[:threshold_boundary], label='recall')

    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))

    plt.xlabel('Threshold value')
    plt.ylabel('Precision and Recall value')
    plt.legend()
    plt.grid()
    plt.show()