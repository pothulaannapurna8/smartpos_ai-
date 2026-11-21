"""Train an experimental image-based GOOD/BAD classifier from Roboflow YOLOv8 data.

Run: python train_image_posture.py --dataset path/to/unzipped-dataset
This model is NOT a medical posture assessment and must be tested on new people.
"""
import argparse
import json
import re
from pathlib import Path
import cv2
import joblib
import numpy as np
from smartpos.image_features import IMAGE_FEATURE_COUNT, image_features
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def load_dataset(root):
    X, y, groups, filenames = [], [], [], []
    for split in ('train', 'valid', 'test'):
        for img in sorted((root / split / 'images').glob('*')):
            if img.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            lab = root / split / 'labels' / (img.stem + '.txt')
            if not lab.exists():
                continue
            labels = [int(line.split()[0]) for line in lab.read_text().splitlines() if line.strip()]
            if not labels or len(set(labels)) != 1 or labels[0] not in (0, 1):
                continue
            frame = cv2.imread(str(img))
            if frame is None:
                continue
            # Group near-adjacent video frames (10-second buckets) to limit leakage.
            match = re.search(r'WIN_(\d{8})_(\d{2})_(\d{2})_(\d{2})', img.name)
            group = (match.group(1) + match.group(2) + match.group(3)
                     + str(int(match.group(4)) // 10)) if match else img.stem.split('.rf.')[0]
            X.append(image_features(frame))
            y.append(labels[0]); groups.append(group); filenames.append(img.name)
    return np.asarray(X), np.asarray(y), np.asarray(groups), filenames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=Path('image_posture_model.joblib'))
    args = ap.parse_args()
    X, y, groups, names = load_dataset(args.dataset)
    if len(np.unique(y)) != 2:
        raise RuntimeError('Both GOOD and BAD classes are required')
    candidate = None
    for seed in range(100):
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed).split(X, y, groups))
        if len(np.unique(y[train])) == 2 and len(np.unique(y[test])) == 2:
            candidate = (train, test, seed)
            break
    if candidate is None:
        raise RuntimeError('Cannot form a grouped evaluation split containing both classes')
    train, test, seed = candidate
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, class_weight='balanced', C=.01))
    model.fit(X[train], y[train])
    pred = model.predict(X[test])
    metrics = {
        'description': 'Experimental HOG + logistic regression image classifier',
        'class_mapping': {'0': 'BAD', '1': 'GOOD'},
        'images': int(len(X)), 'train_images': int(len(train)), 'test_images': int(len(test)),
        'train_counts': {str(k): int(sum(y[train] == k)) for k in (0, 1)},
        'test_counts': {str(k): int(sum(y[test] == k)) for k in (0, 1)},
        'grouped_split_seed': seed,
        'grouped_split_accuracy': float(accuracy_score(y[test], pred)),
        'grouped_split_balanced_accuracy': float(balanced_accuracy_score(y[test], pred)),
        'confusion_matrix_bad_good': confusion_matrix(y[test], pred, labels=[0, 1]).tolist(),
        'class_report': classification_report(
            y[test],
            pred,
            labels=[0, 1],
            target_names=['BAD', 'GOOD'],
            output_dict=True,
            zero_division=0,
        ),
        'warning': 'Same recording/session; not independent-person or webcam accuracy. Do not claim clinical validity.',
    }
    # Refit on all data only after evaluating the held-out groups.
    model.fit(X, y)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'pipeline': model, 'kind': 'hog_image_binary_v1', 'labels': ['BAD', 'GOOD']}, args.output)
    (args.output.parent / 'image_posture_metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
