"""Same fixed learner profiles for all RQ contributions."""
from research.studies.compare_models import model_for


def build_classifier(name, training_images):
    """Project-profile SVM/LightGBM, without E/J or trainable feature scaling."""
    if name not in ('svm', 'lightgbm'):
        raise ValueError(name)
    return model_for(name, False, training_images)
