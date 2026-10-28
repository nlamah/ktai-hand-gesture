"""Small sklearn MLP with train-only class balancing and no random inner holdout."""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight


class BalancedMLPClassifier(MLPClassifier):
    """Match the SVM's balanced class policy using sklearn's weighted MLP fit."""

    class_weight = 'balanced'

    def fit(self, X, y, sample_weight=None):
        targets = np.asarray(y)
        if targets.ndim != 1:
            raise ValueError('The letter MLP requires one class label per photo.')
        weights = compute_sample_weight('balanced', targets)
        classes, counts = np.unique(targets, return_counts=True)
        self.training_class_counts_ = dict(zip(map(str, classes), map(int, counts)))
        self.class_balance_weights_ = {str(c): float(len(targets) / (len(classes) * n)) for c, n in zip(classes, counts)}
        if sample_weight is not None:
            weights = weights * np.asarray(sample_weight)
        return super().fit(X, y, sample_weight=weights)


def mlp_parameters(seed):
    return dict(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                alpha=.0001, batch_size='auto', learning_rate_init=.001,
                max_iter=300, tol=.0001, n_iter_no_change=20,
                early_stopping=False, random_state=seed)


def training_diagnostics(estimator):
    classifier = estimator.steps[-1][1] if hasattr(estimator, 'steps') else estimator
    result = dict(class_weight_policy=getattr(classifier, 'class_weight', None))
    if isinstance(classifier, MLPClassifier):
        result.update(optimizer_iterations=int(classifier.n_iter_), optimizer_max_iter=classifier.max_iter,
            iteration_limit_reached=classifier.n_iter_ >= classifier.max_iter,
            training_loss=float(classifier.loss_), early_stopping=classifier.early_stopping,
            training_class_counts=classifier.training_class_counts_,
            class_balance_weights=classifier.class_balance_weights_)
    return result
