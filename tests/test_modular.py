from pathlib import Path
from io import BytesIO
import json,sys,tempfile,unittest
from unittest.mock import patch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gesturelab.datasets.alphabet import initialize_alphabet,LETTERS,ALPHABET_IMAGES
from gesturelab.datasets.data import read_manifest,add_image,atomic_json
from gesturelab.features.audit import save_annotation,evaluate_annotations
from gesturelab.features.finger_features import FINGERS

class ModularTests(unittest.TestCase):
    def test_predefined_alphabet_is_reference_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=initialize_alphabet(tmp)
            initialize_alphabet(tmp)
            self.assertFalse((p/'labels.json').exists())
            self.assertEqual(len(LETTERS),26)
            self.assertEqual(LETTERS,list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
            self.assertEqual(len(read_manifest(p)['samples']),0)
            for letter in LETTERS:
                copied=p/'references/ngt_alphabet'/f'{letter}.png'
                self.assertEqual(copied.read_bytes(),(ALPHABET_IMAGES/f'{letter}.png').read_bytes())

    def test_human_precision_counts_false_positives_and_missed_detections(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Human truth: extended, extended, bent, extended. Predictions: extended, extended, extended, no hand.
            for index,truth in enumerate(['extended','extended','bent','extended']):
                data=BytesIO();Image.new('RGB',(20,20),(30+index,50,70)).save(data,format='PNG')
                sample,_=add_image(tmp,data.getvalue(),'photo.png','A','s1')
                states={f:'unscorable' for f in FINGERS};states['index']=truth
                save_annotation(tmp,sample['id'],states)
                m=read_manifest(tmp);m['samples'][-1]['status']='no_hand' if index==3 else 'ok';atomic_json(Path(tmp)/'manifest.json',m)
            features={f'geom_{f}_{state}':float(state=='extended') for f in FINGERS for state in ['extended','bent']}
            with patch('gesturelab.features.audit.features_for_sample',return_value=features):
                table,records,meta=evaluate_annotations(tmp)
            row=table[(table.finger=='index')&(table.target=='extended')].iloc[0]
            self.assertEqual((row.true_positives,row.false_positives,row.false_negatives),(2,1,1))
            self.assertAlmostEqual(row.precision,2/3);self.assertAlmostEqual(row.recall,2/3)
            self.assertEqual(meta['unscorable_skipped'],16);self.assertEqual(meta['coverage'],.75)
            self.assertTrue(table[(table.finger=='thumb')].precision.isna().all())

if __name__=='__main__':unittest.main()
