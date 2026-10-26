"""Complete modular flow on explicitly synthetic data; no real accuracy claim."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.demo import make_demo
from gesturelab.training.experiment_plan import prepare_experiment, load_experiment
from gesturelab.training.fitting import train_variant
from gesturelab.training.comparison import compare_variants
from gesturelab.datasets.data import read_manifest,atomic_json

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default='examples/modular_demo')
if __name__=='__main__':
    root=Path(p.parse_args().out).resolve()
    if root.exists():p.exit(1,'Use a new directory for the software test.\n')
    project=make_demo(root/'dataset')
    exp=prepare_experiment(project,root/'experiment',folds=3)
    a=train_variant(exp,'without_features');b=train_variant(exp,'with_features')
    info_a=json.loads((a/'training.json').read_text());info_b=json.loads((b/'training.json').read_text())
    assert (info_a['n_features'],info_b['n_features'])==(126,207)
    assert info_a['split_sha256']==info_b['split_sha256']
    cv=compare_variants(exp);final=compare_variants(exp,True)
    assert compare_variants(exp,True)==final
    plan=json.loads((exp/'split.json').read_text())
    assert not set(plan['test_ids'])&set(plan['development_ids'])
    for fold in plan['folds']:
        assert not set(fold['train_ids'])&set(fold['validation_ids'])
        assert not set(fold['train_groups'])&set(fold['validation_groups'])
    result=json.loads((final/'result.json').read_text());assert result['n']==40
    m=read_manifest(project);m['samples'][0]['condition']='changed';atomic_json(project/'manifest.json',m)
    try:
        load_experiment(exp)
        raise AssertionError('Stale dataset not rejected')
    except ValueError as error:
        assert 'Dataset changed' in str(error)
    m['samples'][0]['condition']='baseline';atomic_json(project/'manifest.json',m)
    assert (cv/'report.html').exists() and (final/'comparison.png').exists()
    print('Modular smoke test passed: 126 vs 207 features; shared CV; reserved test; stale-data guard; charts and HTML.')
    print(final/'report.html')
