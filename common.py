import json
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

import os
ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROOT, 'data')
R = os.path.join(ROOT, 'results')
LAB = ['E', 'S', 'C', 'I']

def metrics(y, pred):
    return {'micro_f1': round(float(f1_score(y, pred, average='micro')), 4),
            'macro_f1': round(float(f1_score(y, pred, average='macro')), 4),
            'per_class_f1': dict(zip(LAB, [round(float(v), 4) for v in f1_score(y, pred, average=None, labels=[0, 1, 2, 3])])),
            'confusion': confusion_matrix(y, pred, labels=[0, 1, 2, 3]).tolist()}

def save(name, obj):
    import os; os.makedirs(R, exist_ok=True)
    json.dump(obj, open(f'{R}/{name}.json', 'w'), indent=1)
