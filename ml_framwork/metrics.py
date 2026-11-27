import numpy as np

def accuracy(y_pred:np.ndarray,y_true:np.ndarray)->float:
    return float(np.mean(y_pred==y_true))

def confusion_matrix(y_pred:np.ndarray,y_true:np.ndarray,num_classes):
    cm=np.zeros((num_classes,num_classes),dtype=np.int64)
    for t,p in zip(y_true,y_pred):
        cm[t,p] +=1
    return cm


def f1_macro(y_pred: np.ndarray, y_true: np.ndarray, num_classes: int) -> float:
    f1s = []

    for k in range(num_classes):
        tp = np.sum((y_pred == k) & (y_true == k))
        fp = np.sum((y_pred == k) & (y_true != k))
        fn = np.sum((y_pred != k) & (y_true == k))

        denom = 2 * tp + fp + fn

        if denom == 0:
            f1 = 0.0
        else:
            f1 = (2 * tp) / denom

        f1s.append(f1)

    return float(np.mean(f1s))


def precision_recall(y_pred, y_true, num_classes):
    precisions, recalls = [], []

    for k in range(num_classes):
        tp = np.sum((y_pred == k) & (y_true == k))
        fp = np.sum((y_pred == k) & (y_true != k))
        fn = np.sum((y_pred != k) & (y_true == k))

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)

        precisions.append(precision)
        recalls.append(recall)

    return np.mean(precisions), np.mean(recalls)






