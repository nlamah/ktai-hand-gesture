"""Friendly labels for live models; world joints are distinct from image-coordinate controls."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MODEL_CHOICES=(
    ('reference','Reference',ROOT/'models/asl_shin.joblib'),
    ('angles','World joints',ROOT/'models/asl_shin_angles.joblib'),
    ('tilt','All-letter tilt',ROOT/'models/asl_shin_angles_tilt15.joblib'),
    ('C_tilt','C-only tilt',ROOT/'models/asl_shin_angles_C_tilt15.joblib'),
    ('image_angles','Image + depth',ROOT/'models/asl_shin_angles_2d.joblib'),
)
PRIMARY_KEYS=('reference','angles')
MODEL_HELP={
    'reference':'Shin baseline: distances and directions (820 inputs).',
    'angles':'The same baseline plus 15 world-joint measurements (835 inputs).',
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
    letters=bundle.get('augmentation',{}).get('augmented_letters')
    return None,'C-only tilt' if letters==['C'] else 'Custom model'
