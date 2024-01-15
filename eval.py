import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

def evaluate(y_pred, y_true, metric):
    if metric == 'acc':
        return eval_acc(y_pred, y_true)
    elif metric == 'auc':
        y_true = y_true.unsqueeze(1)
        return eval_rocauc(y_pred, y_true)
    elif metric == 'f1':
        return eval_f1(y_pred, y_true)
    else:
        raise NotImplementedError('The metric is not supported!')


def eval_acc(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


def eval_rocauc(y_pred, y_true):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).detach().cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list) / len(rocauc_list)


def eval_f1(y_pred, y_true):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    f1 = f1_score(y_true, y_pred, average='weighted')
    # macro_f1 = f1_score(y_true, y_pred, average='macro')
    return f1
