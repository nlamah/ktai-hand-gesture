"""Friendly labels for live models; world joints are distinct from image-coordinate controls."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MODEL_CHOICES=(
    ('reference','Reference',ROOT/'models/asl_shin.joblib'),
    ('angles','World joints',ROOT/'models/asl_shin_angles.joblib'),
    ('tilt','All-letter tilt',ROOT/'models/asl_shin_angles_tilt15.joblib'),
    ('C_tilt','C-only tilt',ROOT/'models/asl_shin_angles_C_tilt15.joblib'),
    ('GH_rotation','G/H rotation',ROOT/'models/asl_shin_angles_GH_rotation.joblib'),
    ('image_angles','Image + depth',ROOT/'models/asl_shin_angles_2d.joblib'),
    ('mirror_base','Fresh original',ROOT/'models/asl_shin_angles_mirror_baseline.joblib'),
    ('mirror','Mirror trained',ROOT/'models/asl_shin_angles_mirrored.joblib'),
    ('GH_back','G/H y180 only',ROOT/'models/asl_shin_angles_GH_y180.joblib'),
    ('right_hand','Right normalized',ROOT/'models/asl_shin_angles_right_hand.joblib'),
)
# Keep historical keys and digit shortcuts stable, including the [2]/[9]
# comparison in Appendix E. Older exploratory variants remain in the archive.
PRIMARY_KEYS=('reference','angles','GH_back','mirror_base','mirror','right_hand')
MODEL_HELP={
    'reference':'Shin-basis: afstanden en richtingen (820 inputs).',
    'angles':'Dezelfde basis met 15 wereldgewrichtsmetingen (835 inputs).',
    'GH_back':'Zelfde 835-inputreferentie, met extra G/H-trainingsbeschrijvingen die 180 graden om de verticale camera-as zijn gedraaid. Geen links/rechts-spiegeling.',
    'mirror_base':'Controlemodel op opnieuw gedetecteerde oorspronkelijke beelden; vergelijk hiermee de spiegeltraining.',
    'mirror':'Getraind met oorspronkelijke en horizontaal gespiegelde beelden, elk opnieuw gedetecteerd.',
    'right_hand':'Detecties worden voor de featureberekening naar dezelfde rechterhandoriëntatie genormaliseerd.',
}


def model_shortcut(key):
    index=next(i for i,choice in enumerate(MODEL_CHOICES) if choice[0]==key)
    return str((index+1)%10)


def visible_model_choices(archive=False):
    return MODEL_CHOICES if archive else tuple(next(c for c in MODEL_CHOICES if c[0]==key) for key in PRIMARY_KEYS)


def visible_model_buttons(archive=False):
    return tuple((key,label,path,(12+(i%3)*210,576+(i//3)*48,208+(i%3)*210,620+(i//3)*48))
                 for i,(key,label,path) in enumerate(visible_model_choices(archive)))


MODEL_BUTTONS=visible_model_buttons()


def describe_model(path,bundle):
    p=Path(path).resolve()
    for key,label,model_path in MODEL_CHOICES:
        if model_path.resolve()==p:return key,label
    if bundle.get('hand_normalization')=='detector-right-landmarks-v1':return None,'Right normalized'
    if bundle.get('augmentation',{}).get('version')=='gh-y180-standalone-v1':return None,'G/H y180 only'
    if bundle.get('gh_view_adaptation'):
        version=bundle['gh_view_adaptation'].get('version','')
        return None,'G/H y180' if version.startswith('gh-camera-y180') else 'G/H index roll'
    letters=bundle.get('augmentation',{}).get('augmented_letters')
    if letters==['G','H']:return None,'G/H rotation'
    return None,'C-only tilt' if letters==['C'] else 'Custom model'
