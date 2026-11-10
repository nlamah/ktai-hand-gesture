"""Frozen input encoding and training-only angle preprocessing."""
import csv
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from research.geometry import FINGER_JOINTS
from research.workbook import categorize_workbook_features
LABELS=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
ANGLE_COLUMNS=tuple(f"{joint}_angle_deg" for joint in FINGER_JOINTS)+("thumb_mcp_signed_deviation_deg",)


def encode_angles(measurements):
    """Fourteen internal joint angles and signed thumb-MCP deviation, in degrees."""
    return np.array([np.nan if measurements[key] is None else measurements[key] for key in ANGLE_COLUMNS])

def angle_estimator(angle_start, n_columns):
    # Keep Shin and categorical values unchanged. Fit imputation and scaling of
    # degree-valued inputs on the fitting fold only.
    preprocessing=ColumnTransformer([
        ('base','passthrough',list(range(angle_start))),
        ('angles',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler()),
         list(range(angle_start,n_columns)))],sparse_threshold=0)
    return make_pipeline(preprocessing,SVC(C=10,gamma='scale',kernel='rbf',class_weight='balanced'))

def category_options(feature):
    if feature=='thumb_mcp_shape':return ['B-vormig','not_B-vormig','unknown']
    if feature.startswith('flexion_'):return ['bent','partly_bent','extended','unknown']
    if feature.startswith('spread_'):return ['spread','not_spread','unknown']
    if feature=='crossing_index_middle':return ['suspected_crossed','not_crossed','unknown']
    return ['contact','no_contact','unknown']

def encode_additions(measurements, parameters, references):
    """Image geometry and fixed reference tables only; no sample label input."""
    actual=categorize_workbook_features(measurements,parameters)
    for feature in list(actual):
        if feature.startswith('thumb_tip_') and actual[feature] in ('near','far'):
            actual[feature]='no_contact'
    keys=list(references['A'])
    categories=np.array([float(actual[feature]==value) for feature in keys for value in category_options(feature)])
    scores=[]
    for label in LABELS:
        cells=[(feature,cell['allowed']) for feature,cell in references[label].items() if cell['allowed'] is not None]
        scores.append(sum(actual[feature] in allowed for feature,allowed in cells)/len(cells))
    return categories,np.array(scores)

def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
