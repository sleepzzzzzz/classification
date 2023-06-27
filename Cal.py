import numpy as np
import torch

from sklearn.metrics import roc_auc_score,\
    precision_recall_curve, \
    auc, \
    f1_score,\
    matthews_corrcoef,\
    recall_score,\
    precision_score,\
    accuracy_score,\
    balanced_accuracy_score,\
    confusion_matrix


__all__ = ['Meter']


class Meter(object):
    def __init__(self, mean=None, std=None):
        self.mask = []
        self.y_pred = []
        self.y_true = []

        if (mean is not None) and (std is not None):
            self.mean = mean.cpu()
            self.std = std.cpu()
        else:
            self.mean = None
            self.std = None

    def update(self, y_pred, y_true, mask=None):
        """Update for the result of an iteration

        Parameters
        ----------
        y_pred : float32 tensor
            Predicted labels with shape ``(B, T)``,
            ``B`` for number of graphs in the batch and ``T`` for the number of tasks
        y_true : float32 tensor
            Ground truth labels with shape ``(B, T)``
        mask : None or float32 tensor
            Binary mask indicating the existence of ground truth labels with
            shape ``(B, T)``. If None, we assume that all labels exist and create
            a one-tensor for placeholder.
        """
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        if mask is None:
            self.mask.append(torch.ones(self.y_pred[-1].shape))
        else:
            self.mask.append(mask.detach().cpu())

    def _finalize(self):
        """Prepare for evaluation.

        If normalization was performed on the ground truth labels during training,
        we need to undo the normalization on the predicted labels.

        Returns
        -------
        mask : float32 tensor
            Binary mask indicating the existence of ground
            truth labels with shape (B, T), B for batch size
            and T for the number of tasks
        y_pred : float32 tensor
            Predicted labels with shape (B, T)
        y_true : float32 tensor
            Ground truth labels with shape (B, T)
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)

        if (self.mean is not None) and (self.std is not None):
            # To compensate for the imbalance between labels during training,
            # we normalize the ground truth labels with training mean and std.
            # We need to undo that for evaluation.
            y_pred = y_pred * self.std + self.mean

        return mask, y_pred, y_true

    def _reduce_scores(self, scores, reduction='none'):
        """Finalize the scores to return.

        Parameters
        ----------
        scores : list of float
            Scores for all tasks.
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        if reduction == 'none':
            return scores
        elif reduction == 'mean':
            return np.mean(scores)
        elif reduction == 'sum':
            return np.sum(scores)
        else:
            raise ValueError(
                "Expect reduction to be 'none', 'mean' or 'sum', got {}".format(reduction))

    def multilabel_score(self, score_func, reduction='none'):
        """Evaluate for multi-label prediction.

        Parameters
        ----------
        score_func : callable
            A score function that takes task-specific ground truth and predicted labels as
            input and return a float as the score. The labels are in the form of 1D tensor.
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        mask, y_pred, y_true = self._finalize()
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            task_score = score_func(task_y_true, task_y_pred)
            if task_score is not None:
                scores.append(task_score)
        return self._reduce_scores(scores, reduction)

    def roc_auc_score(self, reduction='none'):
        """Compute the area under the receiver operating characteristic curve (roc-auc score)
        for binary classification.

        ROC-AUC scores are not well-defined in cases where labels for a task have one single
        class only (e.g. positive labels only or negative labels only). In this case we will
        simply ignore this task and print a warning message.

        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks.

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        # Todo: This function only supports binary classification and we may need
        #  to support categorical classes.
        assert (self.mean is None) and (self.std is None), \
            'Label normalization should not be performed for binary classification.'

        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                print('Warning: Only one class {} present in y_true for a task. '
                      'ROC AUC score is not defined in that case.'.format(y_true[0]))
                return None
            else:
                return roc_auc_score(y_true.long().numpy(), torch.sigmoid(y_pred).numpy())

        return self.multilabel_score(score, reduction)

    def pr_auc_score(self, reduction='none'):
        """Compute the area under the precision-recall curve (pr-auc score)
        for binary classification.

        PR-AUC scores are not well-defined in cases where labels for a task have one single
        class only (e.g. positive labels only or negative labels only). In this case, we will
        simply ignore this task and print a warning message.

        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks.

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        assert (self.mean is None) and (self.std is None), \
            'Label normalization should not be performed for binary classification.'

        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                print('Warning: Only one class {} present in y_true for a task. '
                      'PR AUC score is not defined in that case.'.format(y_true[0]))
                return None
            else:
                precision, recall, _ = precision_recall_curve(
                    y_true.long().numpy(), torch.sigmoid(y_pred).numpy())
                return auc(recall, precision)

        return self.multilabel_score(score, reduction)

    def start_f1_score(self, reduction='None'):
        def score(y_true, y_pred):
            y_pred = torch.sigmoid(y_pred).round()
            return f1_score(y_true, y_pred)

        return self.multilabel_score(score, reduction)

    def start_mcc_score(self, reduction='None'):
        def score(y_true, y_pred):
            y_pred = torch.sigmoid(y_pred).round()
            return matthews_corrcoef(y_true, y_pred)

        return self.multilabel_score(score, reduction)

    def start_recall_score(self, reduction='None'):
        def score(y_true, y_pred):
            y_pred = torch.sigmoid(y_pred).round()
            return recall_score(y_true, y_pred)

        return self.multilabel_score(score, reduction)

    def start_precision_score(self, reduction='None'):
        def score(y_true, y_pred):
            y_pred = torch.sigmoid(y_pred).round()
            return precision_score(y_true, y_pred)

        return self.multilabel_score(score, reduction)

    def start_accuracy_score(self, reduction='None'):
        def score(y_true, y_pred):
            y_pred = torch.sigmoid(y_pred).round()
            return accuracy_score(y_true, y_pred)

        return self.multilabel_score(score, reduction)

    def start_balanced_accuracy_score(self, reduction='None'):
        def score(y_true, y_pred):
            y_pred = torch.sigmoid(y_pred).round()
            return balanced_accuracy_score(y_true, y_pred)

        return self.multilabel_score(score, reduction)

    def start_confusion_matrix(self,reduction = 'None'):
        def score(y_true, y_pred):
            y_pred = torch.sigmoid(y_pred).round()
            return confusion_matrix(y_true, y_pred).tolist()
        return self.multilabel_score(score, reduction)


    def compute_metric(self, metric_name, reduction='none'):
        """Compute metric based on metric name.

        Parameters
        ----------
        metric_name : str
            * ``'roc_auc_score'``: compute roc-auc score
            * ``'pr_auc_score'``: compute pr-auc score
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """

        if metric_name == 'roc_auc_score':
            return self.roc_auc_score(reduction)
        elif metric_name == 'pr_auc_score':
            return self.pr_auc_score(reduction)
        elif metric_name == 'f1_score':
            return self.start_f1_score(reduction)
        elif metric_name == 'mcc':
            return self.start_mcc_score(reduction)
        elif metric_name == 'recall':
            return self.start_recall_score(reduction)
        elif metric_name == 'precision':
            return self.start_precision_score(reduction)
        elif metric_name == 'acc':
            return self.start_accuracy_score(reduction)

        elif metric_name == 'all_classification_results':
            return self.roc_auc_score(reduction), \
                self.pr_auc_score(reduction), \
                self.start_mcc_score(reduction), \
                self.start_f1_score(reduction), \
                self.start_recall_score(reduction), \
                self.start_precision_score(reduction), \
                self.start_accuracy_score(reduction),\
                self.start_balanced_accuracy_score(reduction),\
                self.start_confusion_matrix(reduction)

        else:
            raise ValueError('Expect metric_name to be "f1_score" or "recall" or "mcc" or "acc" or "precision" '
                             'or "roc_auc_score" or "pr_auc", got {}'.format(metric_name))
