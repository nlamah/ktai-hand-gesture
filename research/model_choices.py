"""Friendly labels for all deployed live comparison models."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MODEL_CHOICES=(
    ('reference','Referentie',ROOT/'models/asl_shin.joblib'),
    ('angles','Vingerhoeken',ROOT/'models/asl_shin_angles.joblib'),
    ('tilt','Kanteltraining alle',ROOT/'models/asl_shin_angles_tilt15.joblib'),
    ('C_tilt','C-kantelproef',ROOT/'models/asl_shin_angles_C_tilt15.joblib'),
    ('GH_rotation','G/H-draaitraining',ROOT/'models/asl_shin_angles_GH_rotation.joblib'),
    ('image_angles','Beeld + diepte',ROOT/'models/asl_shin_angles_2d.joblib'),
)
MODEL_BUTTONS=tuple((key,label,path,(12+(i%3)*210,576+(i//3)*48,208+(i%3)*210,620+(i//3)*48))
                    for i,(key,label,path) in enumerate(MODEL_CHOICES))


def describe_model(path,bundle):
    p=Path(path).resolve()
    for key,label,model_path in MODEL_CHOICES:
        if model_path.resolve()==p:return key,label
    letters=bundle.get('augmentation',{}).get('augmented_letters')
    if letters==['G','H']:return None,'G/H-draaitraining'
    return None,'C-kantelproef' if letters==['C'] else 'Eigen model'
