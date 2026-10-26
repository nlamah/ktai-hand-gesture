"""Identical grouped folds, fixed baselines, feature ablations and a separate final test."""
from datetime import datetime, timezone
from dataclasses import dataclass
import json
from pathlib import Path
import time
import textwrap
import uuid
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import sklearn

from gesturelab.datasets.data import analysis_table, read_manifest, fingerprint, atomic_json
from gesturelab.features.inspection import FEATURE_VERSION, feature_columns
from gesturelab.training.rules import UNKNOWN, predict_rule, validate_rules

MODEL_NAMES = {"knn":"k-NN", "svm":"SVM (RBF)", "logistic":"Logistic regression",
               "random_forest":"Random forest", "decision_tree":"Decision tree"}
SET_NAMES = {"raw":"Raw image landmarks", "normalized":"Normalized image landmarks",
             "geometry":"Finger features", "combined":"Image landmarks + finger features", "none":"No features"}


def estimator(name, n_train=100, seed=42):
    if name=="knn":
        return make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=min(5,n_train),weights="distance"))
    if name=="svm":
        return make_pipeline(StandardScaler(),SVC(C=10,gamma="scale",kernel="rbf",class_weight="balanced"))
    if name=="logistic":
        return make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,class_weight="balanced",random_state=seed))
    if name=="random_forest":
        return RandomForestClassifier(n_estimators=150,min_samples_leaf=2,class_weight="balanced",random_state=seed,n_jobs=-1)
    if name=="decision_tree":
        return DecisionTreeClassifier(max_depth=6,min_samples_leaf=2,class_weight="balanced",random_state=seed)
    if name=="dummy":
        return DummyClassifier(strategy="most_frequent")
    raise ValueError(f"Unknown model: {name}")


def groups_for(df, grouping):
    if grouping=="participant":
        return df["participant"].astype(str).to_numpy()
    if grouping=="session":
        return np.array([json.dumps([str(a),str(b)]) for a,b in zip(df.participant,df.session)])
    raise ValueError("Group by session or participant.")


def grouped_folds(df, n_splits=3, grouping="session", seed=42):
    if df.empty or df.label.nunique()<2:
        raise ValueError("At least two labels with successful detections are required.")
    y=df.label.astype(str).to_numpy()
    groups=groups_for(df,grouping)
    counts=pd.DataFrame({"label":y,"group":groups}).groupby("label")["group"].nunique()
    if not isinstance(n_splits,int) or n_splits<2:
        raise ValueError("Choose at least two folds.")
    if int(counts.min())<n_splits:
        unit="sessions" if grouping=="session" else "participants"
        photos=df.groupby("label").size()
        details="; ".join(f"{label}: {photos[label]} photos, {count} {unit}" for label,count in counts.items())
        raise ValueError(f"For {n_splits} folds each label must occur in at least {n_splits} different groups ({unit}). {details}. Multiple photos with the same session name count as one session. Correct session names in View dataset, record a new round, or choose fewer folds. Two sessions require two folds.")

    classes=set(y)
    for attempt in range(20):
        cv=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=seed+attempt)
        folds=list(cv.split(np.zeros((len(df),1)),y,groups))
        if all(set(y[tr])==classes==set(y[va]) for tr,va in folds):
            for train,valid in folds:
                assert not set(groups[train])&set(groups[valid])
            return folds,groups
    raise ValueError("Cannot create grouped folds with all labels in each fold. Record all gestures in each session.")


def metrics(y,predicted,labels):
    y=np.asarray(y,dtype=str)
    predicted=np.asarray(predicted,dtype=str)
    return {"accuracy":float(accuracy_score(y,predicted)),
            "macro_f1":float(f1_score(y,predicted,labels=labels,average="macro",zero_division=0)),
            "balanced_accuracy":float(recall_score(y,predicted,labels=labels,average="macro",zero_division=0)),
            "coverage":float(np.mean(np.isin(predicted,labels))), "n":len(y)}


def title_note(synthetic):
    return "SYNTHETIC SOFTWARE TEST — not a real performance measurement" if synthetic else "Grouped cross-validation — development data"


def heatmap(table, path, title, vmin=0, vmax=1, annotate=True):
    if table.empty:
        return
    fig,ax=plt.subplots(figsize=(max(6,len(table.columns)*1.15),max(3,len(table)*.38+1.8)))
    data=table.to_numpy(dtype=float)
    im=ax.imshow(data,aspect="auto",cmap="Blues",vmin=vmin,vmax=vmax)
    ax.set_xticks(range(len(table.columns)),table.columns,rotation=30,ha="right")
    ax.set_yticks(range(len(table.index)),table.index)
    wrapped="\n".join(textwrap.fill(line,width=65) for line in title.splitlines())
    ax.set_title(wrapped,pad=18,fontsize=10)
    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if np.isfinite(data[i,j]):
                    ax.text(j,i,f"{data[i,j]:.2f}",ha="center",va="center",fontsize=8,
                            color="white" if data[i,j]>.6 else "#17263b")
    fig.colorbar(im,ax=ax,fraction=.03,pad=.025)
    fig.tight_layout()
    fig.savefig(path,dpi=160,bbox_inches="tight")
    plt.close(fig)


def save_confusion(y,predicted,labels,directory,name,title):
    predicted=[p if p in labels else UNKNOWN for p in predicted]
    order=list(labels)+[UNKNOWN]
    counts=confusion_matrix(y,predicted,labels=order)[:len(labels),:]
    table=pd.DataFrame(counts,index=labels,columns=order)
    table.to_csv(directory/f"confusion_{name}.csv")
    norm=table.div(table.sum(axis=1).replace(0,1),axis=0)
    norm.columns=["unknown" if c==UNKNOWN else c for c in norm.columns]
    heatmap(norm,directory/f"confusion_{name}.png",title+"\nRows: actual · columns: predicted")


def _draw_model_scores(summary, directory, note):
    ordered = summary.sort_values('macro_f1_mean')
    fig, ax = plt.subplots(figsize=(10, max(4, len(ordered) * .38 + 1.8)))
    labels = [f"{row.model_name} | {'Configured conditions' if row.model == 'rule_based' else SET_NAMES.get(row.feature_set, row.feature_set)}"
              for row in ordered.itertuples()]
    ax.barh(labels, ordered.macro_f1_mean, xerr=ordered.macro_f1_std.fillna(0), capsize=3)
    ax.set_xlim(0, 1.06)
    ax.set_xlabel('Macro-F1 (0–1); error bars show variation across folds, not a confidence interval')
    ax.set_title(note, fontsize=10)
    fig.tight_layout()
    fig.savefig(directory / 'comparison.png', dpi=160)
    plt.close(fig)


def _draw_feature_effect(summary, directory, note):
    ml = summary[summary.model.isin(MODEL_NAMES)]
    effect = ml.pivot(index='model_name', columns='feature_set', values='macro_f1_mean')
    effect = effect.reindex(columns=[name for name in ['raw', 'normalized', 'geometry', 'combined'] if name in effect.columns])
    effect = effect.rename(columns={'raw': 'Raw', 'normalized': 'Normalized', 'geometry': 'Finger features', 'combined': 'Combined'})
    heatmap(effect, directory / 'feature_ablation.png', 'Effect of feature set · ' + note)


def _draw_label_condition_reports(predictions, labels, directory, note):
    recalls, conditions = [], []
    for candidate, part in predictions.groupby('candidate'):
        recalls.append({'candidate': candidate, **dict(zip(labels, recall_score(
            part.label, part.prediction, labels=labels, average=None, zero_division=0)))})
        for condition, subset in part.groupby('condition'):
            present = sorted(subset.label.unique())
            conditions.append({'candidate': candidate, 'condition': condition,
                               **metrics(subset.label, subset.prediction, present)})
        save_confusion(part.label, part.prediction, labels, directory, candidate, candidate + ' · ' + note)
    recall_table = pd.DataFrame(recalls).set_index('candidate')
    recall_table.to_csv(directory / 'recall_per_label.csv')
    heatmap(recall_table, directory / 'recall_per_label.png', 'Recall per gesture · ' + note)
    condition_table = pd.DataFrame(conditions)
    condition_table.to_csv(directory / 'conditions.csv', index=False)
    heatmap(condition_table.pivot(index='candidate', columns='condition', values='macro_f1'),
            directory / 'conditions.png', 'By condition (labels with samples) · ' + note)


def draw_comparison(summary, predictions, labels, directory, synthetic):
    note = title_note(synthetic)
    _draw_model_scores(summary, directory, note)
    _draw_feature_effect(summary, directory, note)
    _draw_label_condition_reports(predictions, labels, directory, note)


def comparison_readiness(project,grouping="session"):
    """Count photographs separately from independent usable development groups."""
    records=read_manifest(project)["samples"]
    rows=[]
    for label in sorted({s["label"] for s in records}):
        part=[s for s in records if s["label"]==label]
        dev=[s for s in part if s["included"] and s["role"]=="development"]
        valid=[s for s in dev if s["status"]=="ok"]
        groups={s["participant"] if grouping=="participant" else (s["participant"],s["session"]) for s in valid}
        rows.append({"label":label,"photos_total":len(part),"development_usable":len(valid),
                     "groups":len(groups),"pending":sum(s["status"]=="pending" for s in dev),
                     "failed":sum(s["status"] not in ("ok","pending") for s in dev),
                     "test_photos":sum(s["included"] and s["role"]=="test" for s in part)})
    return pd.DataFrame(rows,columns=["label","photos_total","development_usable","groups","pending","failed","test_photos"])


@dataclass
class _ComparisonContext:
    project: Path
    all_data: pd.DataFrame
    development: pd.DataFrame
    test: pd.DataFrame
    folds: list
    groups: np.ndarray
    selected_labels: object
    labels: list
    rules: dict
    synthetic: bool
    grouping: str
    n_splits: int
    seed: int


def _validate_video_groups(project, grouping):
    video_groups = {}
    for sample in read_manifest(project)['samples']:
        if sample['included'] and sample.get('provenance', {}).get('type') == 'video_frame':
            group = sample['participant'] if grouping == 'participant' else (sample['participant'], sample['session'])
            video_groups.setdefault(sample['provenance']['source_sha256'], set()).add((group, sample['role']))
    if any(len(group) > 1 for group in video_groups.values()):
        raise ValueError('Frames from one source video span groups/roles. Keep the source video together.')


def _comparison_dataset(project, labels):
    all_data = analysis_table(project)
    if all_data.empty:
        raise ValueError('First analyze photos with hand detection.')
    development = all_data[all_data.role == 'development'].reset_index(drop=True)
    selected = sorted(set(labels)) if labels is not None else None
    if selected is not None:
        if set(selected) - set(development.label):
            raise ValueError('Not every selected label has usable development photos. Analyze the photos first.')
        development = development[development.label.isin(selected)].reset_index(drop=True)
    return all_data, development, selected


def _comparison_options(models, feature_sets):
    models = list(MODEL_NAMES) if models is None else list(models)
    feature_sets = ['raw', 'normalized', 'geometry', 'combined'] if feature_sets is None else list(feature_sets)
    if not models or not feature_sets:
        raise ValueError('Choose at least one model and feature set.')
    for model in models:
        if model not in MODEL_NAMES:
            raise ValueError(f'Unknown model {model}.')
    return models, feature_sets


def _comparison_context(project, models, feature_sets, n_splits, grouping, seed, labels):
    _validate_video_groups(project, grouping)
    all_data, development, selected = _comparison_dataset(project, labels)
    folds, groups = grouped_folds(development, n_splits, grouping, seed)
    test = all_data[all_data.role == 'test']
    if len(test) and set(groups) & set(groups_for(test, grouping)):
        raise ValueError('The same group occurs in development and test. Adjust sessions/participants or roles.')
    models, feature_sets = _comparison_options(models, feature_sets)
    rules = validate_rules(json.loads((project / 'rules.json').read_text()), development.columns)
    labels = sorted(development.label.unique())
    synthetic = any(s.get('provenance', {}).get('type') == 'synthetic' for s in read_manifest(project)['samples'])
    context = _ComparisonContext(project, all_data, development, test, folds, groups, selected,
                                 labels, rules, synthetic, grouping, n_splits, seed)
    return context, models, feature_sets


def _create_comparison_run(context):
    directory = context.project / 'reports' / (datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
    directory.mkdir(parents=True)
    data, groups = context.development, context.groups
    folds = [{'fold': index, 'train_ids': data.iloc[train].id.tolist(), 'validation_ids': data.iloc[valid].id.tolist(),
              'train_groups': sorted(set(groups[train])), 'validation_groups': sorted(set(groups[valid]))}
             for index, (train, valid) in enumerate(context.folds)]
    atomic_json(directory / 'splits.json', folds)
    atomic_json(directory / 'rules_snapshot.json', context.rules)
    return directory


def _candidate_columns(frame, model, feature_set, rules):
    if model == 'rule_based':
        return sorted({name for rule in rules['rules'] for name in rule['conditions']})
    columns = feature_columns(frame.columns, feature_set) if feature_set != 'none' else ['raw_0_x']
    if not columns:
        raise ValueError('No features for this configuration.')
    return columns


def _fit_comparison_fold(model, train, valid, columns, rules, seed, config):
    caught_messages = []
    if model == 'rule_based':
        fit_seconds = 0.0
        start = time.perf_counter()
        predicted = np.array([predict_rule(row, rules)[0] for row in valid.to_dict('records')])
    else:
        fit = estimator(model, len(train), seed)
        config['parameters'] = str(fit)
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fit.fit(train[columns].to_numpy(), train.label.to_numpy())
        fit_seconds = time.perf_counter() - start
        caught_messages = [str(warning.message) for warning in caught]
        start = time.perf_counter()
        predicted = fit.predict(valid[columns].to_numpy())
    prediction_ms = (time.perf_counter() - start) * 1000 / len(valid)
    return predicted, fit_seconds, prediction_ms, caught_messages


def _evaluate_candidate(context, model, feature_set):
    frame = context.development
    candidate = f'{model}__{feature_set}'
    columns = _candidate_columns(frame, model, feature_set, context.rules)
    config = {'model': model, 'feature_set': feature_set, 'columns': columns}
    scores, predictions = [], []
    for fold, (train_index, valid_index) in enumerate(context.folds):
        train, valid = frame.iloc[train_index], frame.iloc[valid_index]
        predicted, fit_seconds, prediction_ms, caught = _fit_comparison_fold(
            model, train, valid, columns, context.rules, context.seed, config)
        scores.append({'candidate': candidate, 'model': model, 'feature_set': feature_set, 'fold': fold,
                       **metrics(valid.label, predicted, context.labels), 'fit_s': fit_seconds,
                       'predict_ms_per_sample': prediction_ms, 'warnings': ' | '.join(caught)})
        for (_, row), prediction in zip(valid.iterrows(), predicted):
            predictions.append({'candidate': candidate, 'fold': fold, 'id': row.id, 'label': row.label,
                                'prediction': prediction, 'condition': row.condition,
                                'session': row.session, 'participant': row.participant, 'image_path': row.image_path})
    return candidate, config, scores, predictions


def _evaluate_candidates(context, models, feature_sets, progress):
    candidates = [('dummy', 'none'), ('rule_based', 'geometry')] + [(model, features) for model in models for features in feature_sets]
    scores, predictions, configurations = [], [], {}
    for index, (model, feature_set) in enumerate(candidates):
        candidate, config, fold_rows, pred_rows = _evaluate_candidate(context, model, feature_set)
        configurations[candidate] = config
        scores.extend(fold_rows)
        predictions.extend(pred_rows)
        if progress:
            progress((index + 1) / len(candidates))
    return pd.DataFrame(scores), pd.DataFrame(predictions), configurations


def _comparison_summary(fold_scores):
    summary = fold_scores.groupby(['candidate', 'model', 'feature_set'], sort=False).agg(
        macro_f1_mean=('macro_f1', 'mean'), macro_f1_std=('macro_f1', 'std'),
        accuracy_mean=('accuracy', 'mean'), balanced_accuracy_mean=('balanced_accuracy', 'mean'),
        coverage_mean=('coverage', 'mean'), fit_s_mean=('fit_s', 'mean'),
        predict_ms_per_sample=('predict_ms_per_sample', 'mean')).reset_index()
    names = {**MODEL_NAMES, 'dummy': 'Majority baseline', 'rule_based': 'Fixed rules'}
    summary['model_name'] = summary.model.map(names)
    return summary.sort_values('macro_f1_mean', ascending=False, kind='stable')


def _draw_dataset_counts(all_data, directory, synthetic):
    counts = all_data.groupby(['role', 'label']).size().unstack(0, fill_value=0)
    ax = counts.plot.bar(figsize=(8, 4), color=['#397cd5', '#26a18e'])
    ax.set_ylabel('Photos with successful detection')
    ax.set_title('Dataset distribution' + (' — SYNTHETIC' if synthetic else ''))
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    plt.savefig(directory / 'dataset_counts.png', dpi=160)
    plt.close()


def _comparison_manifest(context, configurations):
    manifest = read_manifest(context.project)
    return {
        'status': 'complete', 'feature_version': FEATURE_VERSION, 'dataset_fingerprint': fingerprint(context.project),
        'grouping': context.grouping, 'n_splits': context.n_splits, 'seed': context.seed,
        'labels': context.labels, 'synthetic': context.synthetic, 'selected_labels': context.selected_labels,
        'candidates': configurations, 'sklearn_version': sklearn.__version__,
        'n_development_detected': len(context.development), 'n_test_detected': len(context.test),
        'rules_without_training': True,
        'labels_without_rules': sorted(set(context.labels) - {rule['label'] for rule in context.rules['rules']}),
        'detection_counts': pd.Series([s['status'] for s in manifest['samples'] if s['included']]).value_counts().to_dict(),
        'interpretation': 'CV is development model selection, not a final test. Classifier scores apply to successful hand detections.',
    }


def compare(project, models=None, feature_sets=None, n_splits=3, grouping='session', seed=42, progress=None, labels=None):
    """Compare candidate classifiers using identical grouped development folds."""
    context, models, feature_sets = _comparison_context(Path(project), models, feature_sets, n_splits, grouping, seed, labels)
    directory = _create_comparison_run(context)
    scores, predictions, configurations = _evaluate_candidates(context, models, feature_sets, progress)
    summary = _comparison_summary(scores)
    summary.to_csv(directory / 'summary.csv', index=False)
    scores.to_csv(directory / 'fold_scores.csv', index=False)
    predictions.to_csv(directory / 'predictions.csv', index=False)
    draw_comparison(summary, predictions, context.labels, directory, context.synthetic)
    _draw_dataset_counts(context.all_data, directory, context.synthetic)
    atomic_json(directory / 'run.json', _comparison_manifest(context, configurations))
    return directory


def _selected_final_data(project, info):
    detected = analysis_table(project)
    if info.get('selected_labels') is not None:
        detected = detected[detected.label.isin(info['selected_labels'])]
    train, test = detected[detected.role == 'development'], detected[detected.role == 'test']
    samples = pd.DataFrame([sample for sample in read_manifest(project)['samples'] if sample['included']])
    if info.get('selected_labels') is not None:
        samples = samples[samples.label.isin(info['selected_labels'])]
    all_train, all_test = samples[samples.role == 'development'], samples[samples.role == 'test']
    _validate_final_holdout(all_train, all_test, info)
    return train, test, all_test


def _validate_final_holdout(train, test, info):
    if len(test) and set(groups_for(train, info['grouping'])) & set(groups_for(test, info['grouping'])):
        raise ValueError('Development and test groups overlap.')
    if len(test) and set(test.label) - set(info['labels']):
        raise ValueError('The test contains labels without development samples.')
    if len(test) and (test.status == 'pending').any():
        raise ValueError('First analyze all test photos.')


def _fit_selected_candidate(train, config, seed):
    if config['model'] == 'rule_based':
        return None, []
    fit = estimator(config['model'], len(train), seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fit.fit(train[config['columns']].to_numpy(), train.label.to_numpy())
    return fit, [str(warning.message) for warning in caught]


def _write_final_conditions(all_test, results, destination, synthetic):
    conditions = []
    for condition, subset in all_test.groupby('condition'):
        valid = results[results.condition == condition]
        row = {'condition': condition, 'total': len(subset), 'detected': len(valid),
               'end_to_end_accuracy': float(np.sum(valid.label.to_numpy() == valid.prediction.to_numpy()) / len(subset))}
        if len(valid):
            row.update(metrics(valid.label, valid.prediction, sorted(subset.label.unique())))
        conditions.append(row)
    table = pd.DataFrame(conditions)
    table.to_csv(destination / 'test_conditions.csv', index=False)
    heatmap(table.set_index('condition')[['end_to_end_accuracy']], destination / 'test_conditions.png',
            'Final test by condition: detection failures count as errors' + ('\nSYNTHETIC SOFTWARE TEST' if synthetic else ''))


def _evaluate_final_predictions(bundle, test, all_test, destination):
    predicted = predict_bundle(bundle, test.to_dict('records'))
    scores = {'conditional_on_detection': metrics(test.label, predicted, bundle['labels']),
              'end_to_end_accuracy': float(np.sum(test.label.to_numpy() == predicted) / len(all_test)),
              'detection_rate': len(test) / len(all_test)}
    results = test[['id', 'label', 'image_path', 'condition', 'session', 'participant']].copy()
    results['prediction'] = predicted
    results.to_csv(destination / 'test_predictions.csv', index=False)
    save_confusion(test.label, predicted, bundle['labels'], destination, 'final',
                   'Independent test' + (' — SYNTHETIC' if bundle['synthetic'] else ''))
    _write_final_conditions(all_test, results, destination, bundle['synthetic'])
    return scores


def _final_selection_report(bundle, train, test, all_test, caught, destination):
    report = {'candidate': bundle['candidate'], 'synthetic': bundle['synthetic'], 'n_train': len(train),
              'n_test_total': len(all_test), 'n_test_detected': len(test), 'warnings': caught}
    if len(test):
        report.update(_evaluate_final_predictions(bundle, test, all_test, destination))
    elif len(all_test):
        report.update({'detection_rate': 0.0, 'end_to_end_accuracy': 0.0})
    else:
        report['note'] = 'No reserved test set; model trained. CV remains a development result.'
    return report


def finalize(project, run, candidate):
    """Fit one selected candidate and evaluate the reserved holdout once per run."""
    project, run = Path(project), Path(run)
    info = json.loads((run / 'run.json').read_text())
    if info['dataset_fingerprint'] != fingerprint(project):
        raise ValueError('Dataset changed since the comparison. Create a new comparison.')
    if (run / 'final').exists():
        raise ValueError('A final selection has already been saved for this run. View that result.')
    config = info['candidates'][candidate]
    train, test, all_test = _selected_final_data(project, info)
    rules = json.loads((run / 'rules_snapshot.json').read_text())
    fit, caught = _fit_selected_candidate(train, config, info['seed'])
    bundle = {'estimator': fit, 'config': config, 'rules': rules, 'feature_version': FEATURE_VERSION,
              'labels': info['labels'], 'candidate': candidate, 'synthetic': info['synthetic']}
    destination = run / 'final'
    destination.mkdir()
    joblib.dump(bundle, destination / 'model.joblib')
    report = _final_selection_report(bundle, train, test, all_test, caught, destination)
    atomic_json(destination / 'final.json', report)
    return destination


def predict_bundle(bundle,rows):
    if bundle["feature_version"]!=FEATURE_VERSION:
        raise ValueError("Model and feature versions do not match.")
    if bundle["config"]["model"]=="rule_based":
        return np.array([predict_rule(r,bundle["rules"])[0] for r in rows])
    columns=bundle["config"]["columns"]
    return bundle["estimator"].predict(np.array([[r[c] for c in columns] for r in rows],dtype=float))


def train_preview(project,model="svm",feature_set="combined",labels=None):
    """Train development images for live practice; never claim an evaluation score."""
    project=Path(project)
    df=analysis_table(project)
    if df.empty:
        raise ValueError("First analyze photos with hand detection.")
    df=df[df.role=="development"]
    if labels is not None:
        df=df[df.label.isin(labels)]
    if df.label.nunique()<2:
        raise ValueError("A practice model requires at least two labels with usable development photos.")
    if model not in MODEL_NAMES:
        raise ValueError("Choose an available ML method.")
    columns=feature_columns(df.columns,feature_set)
    fit=estimator(model,len(df))
    fit.fit(df[columns].to_numpy(),df.label.to_numpy())
    rules=validate_rules(json.loads((project/"rules.json").read_text()),df.columns)
    bundle={"estimator":fit,"config":{"model":model,"feature_set":feature_set,"columns":columns},
            "rules":rules,"feature_version":FEATURE_VERSION,"labels":sorted(df.label.unique()),
            "candidate":f"{model}__{feature_set}","preview":True,
            "synthetic":any(s.get("provenance",{}).get("type")=="synthetic" for s in read_manifest(project)["samples"])}
    directory=project/"reports"/("preview_"+datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:6])
    directory.mkdir(parents=True)
    joblib.dump(bundle,directory/"model.joblib")
    atomic_json(directory/"preview.json",{"note":"Practice model: trained only, not evaluated. Test photos were not used.",
        "labels":bundle["labels"],"train_ids":df.id.tolist(),"n_train":len(df),"synthetic":bundle["synthetic"],
        "candidate":bundle["candidate"],"dataset_fingerprint":fingerprint(project),"sklearn_version":sklearn.__version__})
    return directory/"model.joblib"
