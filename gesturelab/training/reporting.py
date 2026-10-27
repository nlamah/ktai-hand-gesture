"""Compact readable score tables; full precision and timings remain in the CSV files."""


def summary_html(table):
    names = {
        'classifier': 'Classifier', 'variant': 'Input', 'n_features': 'Features', 'n_train': 'Training photos',
        'accuracy': 'Accuracy', 'macro_f1': 'Macro-F1', 'cv_macro_f1_mean': 'CV F1 mean',
        'cv_macro_f1_std': 'CV F1 SD', 'model_training_seconds': 'Model training (s)',
    }
    columns = [name for name in names if name in table]
    compact = table[columns].rename(columns=names)
    return '<div style="overflow-x:auto">' + compact.to_html(index=False, float_format=lambda x: f'{x:.4f}') + '</div>'
