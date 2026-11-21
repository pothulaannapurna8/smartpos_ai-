from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
import joblib
from posture_inference import JOINTS, features_from_points

root = Path(__file__).resolve().parent
csv = root / 'data.csv'
if not csv.exists():
    raise SystemExit('Place downloaded MultiPosture data.csv beside train_model.py')
df = pd.read_csv(csv)
X = np.vstack([features_from_points({j: (row[j+'_x'],row[j+'_y']) for j in JOINTS}) for _,row in df.iterrows()])
encoder = LabelEncoder()
y = encoder.fit_transform(df.upperbody_label.to_numpy())
groups = df.subject.to_numpy()
# Hold out entire people, not random adjacent frames, to reduce evaluation leakage.
splitter = GroupShuffleSplit(n_splits=1, test_size=3/13, random_state=42)
train, test = next(splitter.split(X,y,groups))
model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(48,24),
                   activation='relu', early_stopping=True, validation_fraction=0.15,
                   n_iter_no_change=18, max_iter=350, random_state=42))
model.fit(X[train],y[train])
pred = model.predict(X[test])
metrics = {
    'source': 'MultiPosture dataset (user supplied)',
    'samples_total': int(len(df)),
    'training_samples': int(len(train)), 'test_samples': int(len(test)),
    'train_subjects': sorted(map(int,set(groups[train]))),
    'test_subjects': sorted(map(int,set(groups[test]))),
    'accuracy': float(accuracy_score(y[test],pred)),
    'balanced_accuracy': float(balanced_accuracy_score(y[test],pred)),
    'class_report': classification_report(y[test],pred,output_dict=True,zero_division=0),
    'confusion_matrix': confusion_matrix(y[test],pred,labels=list(model.classes_)).tolist(),
    'classes': list(encoder.classes_),
    'model': 'StandardScaler + MLPClassifier(48,24)',
    'features': [j+'_'+axis for j in JOINTS for axis in ('x','y')],
    'note': 'Offline evaluation on held-out subjects; not medical validation or live webcam accuracy.'
}
joblib.dump({'pipeline':model,'features':metrics['features'],'labels':list(encoder.classes_),
             'source':'MultiPosture','schema_version':1},root/'posture_model.joblib',compress=3)
(root/'metrics.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps({k:v for k,v in metrics.items() if k not in ('class_report','confusion_matrix')},indent=2))
