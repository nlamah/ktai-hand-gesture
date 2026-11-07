"""Eight controlled D/S/E/J representations and physically grouped permutations."""
from collections import OrderedDict

import numpy as np
from sklearn.pipeline import make_pipeline

from research.inputs import ANGLE_COLUMNS, angle_estimator, category_options
from research.studies.compare_models import model_for

VERSION = 'feature-factorial-dsej-v1'
BLOCKS = {'D':slice(0,190), 'S':slice(190,820), 'E':slice(820,890), 'J':slice(916,931)}
VARIANTS = OrderedDict((key,tuple(key)) for key in ('DS','D','DSE','DE','DSJ','DJ','DSEJ','DEJ'))


def source_columns(variant):
    if variant not in VARIANTS:raise ValueError('Unknown feature-study variant.')
    return np.concatenate([np.arange(BLOCKS[b].start,BLOCKS[b].stop) for b in VARIANTS[variant]])


def matrix(raw,variant):
    values=np.asarray(raw)
    if values.ndim!=2 or values.shape[1]!=931:raise ValueError('Expected the frozen N x 931 source cache.')
    return values[:,source_columns(variant)]


def estimator(name,variant,n_train):
    learner=model_for(name,False,n_train)
    if 'J' not in VARIANTS[variant]:return learner
    width=len(source_columns(variant))
    # Exactly the previous train-only imputation/standardization for the last 15 values.
    return make_pipeline(angle_estimator(width-15,width).steps[0][1],learner)


def category_columns(reference):
    result={};offset=820
    for feature in reference:
        result[feature]=list(range(offset,offset+len(category_options(feature))))
        offset+=len(result[feature])
    if offset!=890 or len(result)!=19:raise ValueError('Expected the 19 checks / 70 one-hot values.')
    return result


def family(feature):
    if feature in ('thumb_mcp_shape','flexion_thumb_ip'):return 'thumb'
    if feature.startswith('flexion_'):return 'finger_flexion'
    if feature.startswith('spread_'):return 'spread'
    if feature.startswith('crossing_'):return 'crossing'
    return 'contact'


def permutation_groups(variant,reference):
    """Blocks in every arm; category checks in E-only arms; families in complete arms.

    Detailed checks deliberately use DSE/DE (without J), so duplicate joint
    information does not automatically mask category reliance. Full arms
    DSEJ/DEJ additionally receive E-family and J-finger diagnostics.
    Every column of a group follows the SAME source-row permutation.
    """
    source=source_columns(variant);local={int(s):i for i,s in enumerate(source)}
    groups={}
    for block in VARIANTS[variant]:
        groups[f'block_{block}']=[local[s] for s in range(BLOCKS[block].start,BLOCKS[block].stop)]
    categories=category_columns(reference)
    if variant in ('DSE','DE'):
        for feature,columns in categories.items():groups['check_'+feature]=[local[s] for s in columns]
    if variant in ('DSEJ','DEJ'):
        for name in ('thumb','finger_flexion','spread','crossing','contact'):
            groups['family_E_'+name]=[local[s] for feature,columns in categories.items()
                                      if family(feature)==name for s in columns]
        for name in ('thumb','index','middle','ring','pinky'):
            groups['family_J_'+name]=[local[916+i] for i,column in enumerate(ANGLE_COLUMNS)
                                     if column.startswith(name+'_')]
    return groups


def permute_group(values,columns,order):
    values=np.asarray(values);order=np.asarray(order)
    if order.shape!=(len(values),) or not np.array_equal(np.sort(order),np.arange(len(values))):
        raise ValueError('Order must permute every validation row exactly once.')
    result=values.copy()
    result[:,columns]=values[order][:,columns]
    return result
