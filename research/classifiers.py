"""Explicit fixed project and paper parameter profiles."""
def classifier_parameters(name,variant=None,profile='project',n_train=100,seed=42):
    if name!='lightgbm': raise ValueError(name)
    p=dict(num_leaves=31,learning_rate=.1,min_child_samples=20,n_estimators=100)
    if profile=='shin2021_asl':p.update(num_leaves=100,min_child_samples=25,n_estimators=14)
    elif profile!='project':raise ValueError(profile)
    return dict(**p,random_state=seed,n_jobs=1,deterministic=True,force_col_wise=True,verbosity=-1)
