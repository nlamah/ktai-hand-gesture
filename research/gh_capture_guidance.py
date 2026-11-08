"""Large English G/H capture instructions and a camera/palm-direction diagram."""
from functools import lru_cache
from math import cos, sin, radians
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

VERSION = 'large-en-gh-guidance-v1'
WIDTH, HEIGHT = 1160, 752
BACKGROUND = '#111923'
TEXT = '#f5f7fb'
MUTED = '#c1cbd7'
PALM = '#ffc36a'
BACK = '#71c9ee'
VIEW_TEXT = {
    'handpalm': ('PALM TOWARDS CAMERA', 'Show your palm to the camera. Your knuckles face away. This is the palm control.', 0),
    'schuin_palm': ('OBLIQUE: PALM VISIBLE', 'Start palm towards camera. Turn about 45 degrees; keep part of the palm visible.', 45),
    'schuin_handrug': ('OBLIQUE: BACK VISIBLE', 'Start back towards camera. Turn about 45 degrees; keep part of the back visible.', 135),
    'handrug': ('BACK TOWARDS CAMERA', 'Show your knuckles and hand back. Your palm faces away from the camera.', 180),
}


@lru_cache(maxsize=24)
def font(size, bold=False):
    names = ('Arial Bold.ttf', 'DejaVuSans-Bold.ttf') if bold else ('Arial.ttf', 'DejaVuSans.ttf')
    candidates = [Path('/System/Library/Fonts/Supplemental')/names[0],
                  Path('/usr/share/fonts/truetype/dejavu')/names[1], *names]
    for path in candidates:
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def block(draw, x, y, text, size=24, bold=False, fill=TEXT, width=460):
    face = font(size, bold)
    for paragraph in text.split('\n'):
        line = ''
        for word in paragraph.split():
            candidate = f'{line} {word}'.strip()
            if line and draw.textlength(candidate, font=face) > width:
                draw.text((x, y), line, font=face, fill=fill)
                y += size+6
                line = word
            else:
                line = candidate
        draw.text((x, y), line, font=face, fill=fill)
        y += size+6
    return y


def instructions(task):
    if task['letter'] not in ('G', 'H') or task['requested_hand'] not in ('right', 'left'):
        raise ValueError('G/H instructions need a valid letter and requested hand.')
    title, view, angle = VIEW_TEXT[task['view']]
    shape = ('Extend your index finger and thumb roughly parallel. Bend the other three fingers.'
             if task['letter'] == 'G' else
             'Extend index and middle fingers together. Bend ring and little fingers; fold the thumb.')
    return {'letter': task['letter'], 'hand': 'RIGHT HAND' if task['requested_hand'] == 'right' else 'LEFT HAND',
            'finger_shape': shape, 'view_title': title, 'view_instruction': view,
            'palm_normal_degrees': angle,
            'diagram_caption': 'Arrow = palm direction, not finger direction.'}


def diagram(draw, x, y, angle):
    """Top-down cross section: arrow is palm normal, never finger direction."""
    cx, cy = x+240, y+48
    a = radians(angle)
    normal = np.array([sin(a), cos(a)])
    tangent = np.array([cos(a), -sin(a)])
    center = np.array([cx, cy])
    ends = [center-60*tangent, center+60*tangent]
    draw.line([tuple(p+6*normal) for p in ends], fill=PALM, width=9)
    draw.line([tuple(p-6*normal) for p in ends], fill=BACK, width=9)
    tip = center+43*normal
    draw.line([tuple(center+13*normal), tuple(tip)], fill=PALM, width=4)
    draw.polygon([tuple(tip), tuple(tip-12*normal+7*tangent), tuple(tip-12*normal-7*tangent)], fill=PALM)
    draw.rounded_rectangle((cx-40, y+105, cx+40, y+131), radius=5, outline=TEXT, width=2)
    draw.ellipse((cx-8, y+110, cx+8, y+126), outline=TEXT, width=2)
    draw.text((cx-37, y+136), 'CAMERA', font=font(17, True), fill=TEXT)
    draw.text((x+5, y+14), 'palm side', font=font(18), fill=PALM)
    draw.text((x+5, y+39), 'hand back', font=font(18), fill=BACK)
    draw.text((x+338, y+17), 'Top view', font=font(18), fill=MUTED)


def render(preview, task, completed, total, prompt=None, notice=None):
    info = instructions(task)
    if preview.shape != (480, 640, 3):
        raise ValueError('Expected the unchanged 640 x 480 camera preview.')
    canvas = Image.new('RGB', (WIDTH, HEIGHT), BACKGROUND)
    canvas.paste(Image.fromarray(preview[:, :, ::-1]), (0, 0))
    draw = ImageDraw.Draw(canvas)
    x = 662
    y = block(draw, x, 12, f"{task['participant']} · Photo {completed+1}/{total} · Attempt {task['repeat']}/5", 21, fill=MUTED)
    y = block(draw, x, y+3, f"Use your {info['hand']}", 27, True)
    y = block(draw, x, y+3, f"LETTER {info['letter']}", 52, True)
    y = block(draw, x, y+6, info['finger_shape'], 24)
    y = block(draw, x, y+18, info['view_title'], 29, True, PALM if task['view'] in ('handpalm','schuin_palm') else BACK)
    y = block(draw, x, y+4, info['view_instruction'], 23)
    diagram(draw, x, y+4, info['palm_normal_degrees'])
    y += 169
    bottom = block(draw, x, y, info['diagram_caption'], 18, fill=MUTED)
    if bottom > HEIGHT-5:
        raise ValueError('G/H instructions do not fit the display.')

    action = 'Take photo [Space]'
    if prompt and 'press Enter to begin' in prompt:
        action = f"{task['participant']}: press Enter to begin"
    elif prompt and 'Intended letter' in prompt:
        action = f"Letter {task['letter']} · Enter: save"
    elif prompt:
        action = prompt
    draw.rounded_rectangle((12, 488, 628, 532), radius=5, fill='#e9edf2')
    block(draw, 26, 496, action, 25, True, '#111923', width=586)
    block(draw, 18, 548, 'Form the letter, then turn the hand', 27, True, width=604)
    block(draw, 18, 591, 'Keep the finger shape and turn your whole hand.\nPointing direction may change as you turn.', 23, width=604)
    message = 'Prediction hidden. Save failed detections too.'
    if notice and 'Check participant' in notice:
        message = 'Check participant and hand. Enter starts the round.'
    elif notice and notice.startswith('Saved:'):
        message = 'Photo saved. Reform the sign for the next attempt.'
    elif notice:
        message = notice
    block(draw, 18, 659, message, 20, fill=MUTED, width=604)
    block(draw, 18, 725, 'Spatie: foto | Enter: save | Q/Esc: pauze', 20, width=604)
    return np.asarray(canvas)[:, :, ::-1].copy()
