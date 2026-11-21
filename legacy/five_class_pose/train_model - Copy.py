"""Compare posture classifiers using fixed, participant-disjoint data splits."""
import argparse
import csv
import json
import platform
import warnings
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from posture_inference import (
    CLASS_LABELS,
    FEATURE_NAMES_BY_DIMENSIONS,
    JOINTS,
    features_from_points,
)


def parse_subjects(value):
    try:
        subjects = tuple(int(item.strip()) for item in value.split(',') if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Subjects must be comma-separated integers') from exc
    if not subjects or len(set(subjects)) != len(subjects):
        raise argparse.ArgumentTypeError('Provide unique subject IDs')
    return subjects


def load_dataset(path):
    with path.open(newline='', encoding='utf-8-sig') as source:
        reader = csv.DictReader(source)
        required = {'upperbody_label', 'subject'} | {
            f'{joint}_{axis}'
            for joint in JOINTS
            for axis in ('x', 'y', 'z')
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f'Dataset is missing required columns: {sorted(missing)}')
        rows = list(reader)

    if not rows:
        raise ValueError('Dataset contains no labeled rows')

    features_2d = []
    features_3d = []
    labels = []
    groups = []
    invalid_rows = []
    for line_number, row in enumerate(rows, start=2):
        try:
            points = {
                joint: tuple(
                    float(row[f'{joint}_{axis}'])
                    for axis in ('x', 'y', 'z')
                )
                for joint in JOINTS
            }
            if not row['upperbody_label'].strip():
                raise ValueError('missing upperbody_label')
            subject = int(row['subject'])
            features_2d.append(features_from_points(points, dimensions=2))
            features_3d.append(features_from_points(points, dimensions=3))
        except (TypeError, ValueError, KeyError, OverflowError) as exc:
            invalid_rows.append(f'line {line_number}: {exc}')
            continue
        labels.append(row['upperbody_label'].strip())
        groups.append(subject)

    if invalid_rows:
        preview = '; '.join(invalid_rows[:5])
        raise ValueError(
            f'{len(invalid_rows)} invalid dataset row(s); no rows were silently '
            f'dropped. First issues: {preview}'
        )

    classes = tuple(sorted(set(labels)))
    if classes != CLASS_LABELS:
        raise ValueError(
            f'Dataset classes must be exactly {list(CLASS_LABELS)}; '
            f'found {list(classes)}'
        )
    return (
        np.vstack(features_2d),
        np.vstack(features_3d),
        np.asarray([CLASS_LABELS.index(label) for label in labels], dtype=int),
        np.asarray(groups, dtype=int),
        rows,
    )


def create_candidates():
    return [
        {
            'name': 'majority_baseline',
            'dimensions': 2,
            'factory': lambda: DummyClassifier(strategy='most_frequent'),
            'balanced_mlp': False,
        },
        {
            'name': 'current_mlp_xy_48x24',
            'dimensions': 2,
            'factory': lambda: make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(48, 24),
                    activation='relu',
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=18,
                    max_iter=350,
                    random_state=42,
                ),
            ),
            'balanced_mlp': False,
        },
        {
            'name': 'mlp_xyz_48x24_alpha_0.01',
            'dimensions': 3,
            'factory': lambda: make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(48, 24),
                    activation='relu',
                    alpha=0.01,
                    early_stopping=False,
                    max_iter=600,
                    random_state=42,
                ),
            ),
            'balanced_mlp': True,
        },
        {
            'name': 'mlp_xyz_96x48_alpha_0.1',
            'dimensions': 3,
            'factory': lambda: make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(96, 48),
                    activation='relu',
                    alpha=0.1,
                    early_stopping=False,
                    max_iter=600,
                    random_state=42,
                ),
            ),
            'balanced_mlp': True,
        },
        {
            'name': 'random_forest_xyz_depth_12',
            'dimensions': 3,
            'factory': lambda: RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=2,
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=-1,
            ),
            'balanced_mlp': False,
        },
        {
            'name': 'random_forest_xyz_unbounded',
            'dimensions': 3,
            'factory': lambda: RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=-1,
            ),
            'balanced_mlp': False,
        },
        {
            'name': 'svc_rbf_xyz_c_0.5',
            'dimensions': 3,
            'factory': lambda: make_pipeline(
                StandardScaler(),
                SVC(
                    C=0.5,
                    kernel='rbf',
                    class_weight='balanced',
                    probability=True,
                    random_state=42,
                ),
            ),
            'balanced_mlp': False,
        },
        {
            'name': 'svc_rbf_xyz_c_2',
            'dimensions': 3,
            'factory': lambda: make_pipeline(
                StandardScaler(),
                SVC(
                    C=2,
                    kernel='rbf',
                    class_weight='balanced',
                    probability=True,
                    random_state=42,
                ),
            ),
            'balanced_mlp': False,
        },
        {
            'name': 'svc_rbf_xyz_c_8',
            'dimensions': 3,
            'factory': lambda: make_pipeline(
                StandardScaler(),
                SVC(
                    C=8,
                    kernel='rbf',
                    class_weight='balanced',
                    probability=True,
                    random_state=42,
                ),
            ),
            'balanced_mlp': False,
        },
    ]


def fit_model(model, candidate, features, labels):
    if candidate['balanced_mlp']:
        sample_weights = sklearn.utils.class_weight.compute_sample_weight(
            class_weight='balanced',
            y=labels,
        )
        model.fit(features, labels, mlpclassifier__sample_weight=sample_weights)
    else:
        model.fit(features, labels)
    return model


def evaluate(y_true, y_pred):
    labels = list(range(len(CLASS_LABELS)))
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_precision': float(precision_score(
            y_true, y_pred, labels=labels, average='macro', zero_division=0
        )),
        'macro_recall': float(recall_score(
            y_true, y_pred, labels=labels, average='macro', zero_division=0
        )),
        'macro_f1': float(f1_score(
            y_true, y_pred, labels=labels, average='macro', zero_division=0
        )),
        'confusion_matrix': confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist(),
        'per_class': classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=CLASS_LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }


def grouped_cross_validation(candidate, features, labels, groups):
    folds = GroupKFold(n_splits=4)
    metrics = []
    for train_indices, validation_indices in folds.split(features, labels, groups):
        model = candidate['factory']()
        fit_model(
            model,
            candidate,
            features[train_indices],
            labels[train_indices],
        )
        prediction = model.predict(features[validation_indices])
        fold_metrics = evaluate(labels[validation_indices], prediction)
        metrics.append(fold_metrics)
    return {
        'folds': len(metrics),
        'macro_f1_mean': float(np.mean([item['macro_f1'] for item in metrics])),
        'balanced_accuracy_mean': float(
            np.mean([item['balanced_accuracy'] for item in metrics])
        ),
    }


def subject_bootstrap(y_true, y_pred, groups, samples=1000):
    rng = np.random.default_rng(42)
    subjects = np.unique(groups)
    subject_indices = {
        subject: np.flatnonzero(groups == subject)
        for subject in subjects
    }
    estimates = {'accuracy': [], 'macro_f1': []}
    for _ in range(samples):
        selected = rng.choice(subjects, size=len(subjects), replace=True)
        indices = np.concatenate([subject_indices[subject] for subject in selected])
        estimates['accuracy'].append(
            accuracy_score(y_true[indices], y_pred[indices])
        )
        estimates['macro_f1'].append(f1_score(
            y_true[indices],
            y_pred[indices],
            labels=list(range(len(CLASS_LABELS))),
            average='macro',
            zero_division=0,
        ))
    return {
        name: [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]
        for name, values in estimates.items()
    }


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=root.parent / 'data.csv')
    parser.add_argument('--model-out', type=Path, default=root / 'posture_model.joblib')
    parser.add_argument('--metrics-out', type=Path, default=root / 'metrics.json')
    parser.add_argument(
        '--validation-subjects', type=parse_subjects, default=(10, 12),
        help='Comma-separated participant IDs used only for model selection.',
    )
    parser.add_argument(
        '--test-subjects', type=parse_subjects, default=(9, 11, 13),
        help='Comma-separated participant IDs held out from fitting and tuning.',
    )
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(
            f'Dataset not found: {args.data}. Supply MultiPosture data.csv with --data.'
        )
    try:
        X2, X3, y, groups, rows = load_dataset(args.data)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    all_subjects = set(map(int, np.unique(groups)))
    validation_subjects = set(args.validation_subjects)
    test_subjects = set(args.test_subjects)
    if validation_subjects & test_subjects:
        raise SystemExit('Validation and test participant IDs must not overlap.')
    if not (validation_subjects | test_subjects).issubset(all_subjects):
        raise SystemExit('A requested validation or test participant is not in the dataset.')
    train_subjects = all_subjects - validation_subjects - test_subjects
    if len(train_subjects) < 4:
        raise SystemExit('At least four training participants are required for grouped CV.')

    train_indices = np.flatnonzero(np.isin(groups, list(train_subjects)))
    validation_indices = np.flatnonzero(np.isin(groups, list(validation_subjects)))
    test_indices = np.flatnonzero(np.isin(groups, list(test_subjects)))
    if not len(validation_indices) or not len(test_indices):
        raise SystemExit('Validation and test splits must both contain samples.')
    for name, indices in (
        ('training', train_indices),
        ('validation', validation_indices),
        ('test', test_indices),
    ):
        present = set(y[indices])
        if present != set(range(len(CLASS_LABELS))):
            raise SystemExit(
                f'{name.title()} split does not contain every posture class: '
                f'{[CLASS_LABELS[index] for index in sorted(present)]}'
            )

    X_by_dimensions = {2: X2, 3: X3}
    candidates = create_candidates()
    results = []
    for candidate in candidates:
        dimensions = candidate['dimensions']
        features = X_by_dimensions[dimensions]
        cv = grouped_cross_validation(
            candidate,
            features[train_indices],
            y[train_indices],
            groups[train_indices],
        )
        validation_model = candidate['factory']()
        fit_model(
            validation_model,
            candidate,
            features[train_indices],
            y[train_indices],
        )
        validation_prediction = validation_model.predict(features[validation_indices])
        validation_metrics = evaluate(y[validation_indices], validation_prediction)
        results.append({
            'candidate': candidate,
            'cv': cv,
            'validation': validation_metrics,
        })
        print(
            f"{candidate['name']}: validation macro-F1 "
            f"{validation_metrics['macro_f1']:.3f}, "
            f"balanced accuracy {validation_metrics['balanced_accuracy']:.3f}"
        )

    selected = max(
        results,
        key=lambda item: (
            item['validation']['macro_f1'],
            item['validation']['balanced_accuracy'],
        ),
    )
    selected_name = selected['candidate']['name']
    fit_indices = np.concatenate((train_indices, validation_indices))
    final_results = []
    selected_model = None
    selected_test_prediction = None
    for result in results:
        candidate = result['candidate']
        features = X_by_dimensions[candidate['dimensions']]
        model = candidate['factory']()
        fit_model(model, candidate, features[fit_indices], y[fit_indices])
        test_prediction = model.predict(features[test_indices])
        test_metrics = evaluate(y[test_indices], test_prediction)
        final_results.append({
            'name': candidate['name'],
            'dimensions': candidate['dimensions'],
            'feature_order': list(
                FEATURE_NAMES_BY_DIMENSIONS[candidate['dimensions']]
            ),
            'grouped_cv': result['cv'],
            'validation': result['validation'],
            'test': test_metrics,
        })
        if candidate['name'] == selected_name:
            selected_model = model
            selected_test_prediction = test_prediction

    test_groups = groups[test_indices]
    per_subject = {}
    for subject in sorted(test_subjects):
        subject_mask = test_groups == subject
        per_subject[str(subject)] = evaluate(
            y[test_indices][subject_mask],
            selected_test_prediction[subject_mask],
        )

    label_counts = {
        label: int(sum(row['upperbody_label'] == label for row in rows))
        for label in CLASS_LABELS
    }
    subject_counts = {
        str(subject): int(np.count_nonzero(groups == subject))
        for subject in sorted(all_subjects)
    }
    selected_entry = next(
        item for item in final_results if item['name'] == selected_name
    )
    metrics = {
        'source': 'User-supplied MultiPosture data.csv',
        'dataset': {
            'path': str(args.data),
            'samples': len(rows),
            'subjects': sorted(all_subjects),
            'rows_per_subject': subject_counts,
            'class_counts': label_counts,
            'invalid_rows': 0,
            'exact_duplicate_rows': len(rows) - len({
                tuple(row.get(key, '') for key in sorted(row))
                for row in rows
            }),
            'attribution_note': (
                'The supplied CSV and companion README did not include an '
                'upstream URL or license file; verify original terms before '
                'redistributing the dataset or trained artifact.'
            ),
        },
        'split': {
            'training_subjects': sorted(train_subjects),
            'validation_subjects': sorted(validation_subjects),
            'test_subjects': sorted(test_subjects),
            'training_samples': len(train_indices),
            'validation_samples': len(validation_indices),
            'test_samples': len(test_indices),
            'test_is_participant_disjoint_from_current_fit_and_tuning': True,
            'historical_exposure_note': (
                'All participants were present in prior project model '
                'development. This split is independent within this experiment, '
                'not a prospective external replication.'
            ),
        },
        'selection': {
            'metric': 'validation macro-F1, then validation balanced accuracy',
            'selected_model': selected_name,
            'selected_dimensions': selected['candidate']['dimensions'],
            'validation_metrics': selected['validation'],
        },
        'evaluation': {
            'selected_model_test_metrics': selected_entry['test'],
            'selected_model_test_subject_metrics': per_subject,
            'subject_bootstrap_95_percentile_intervals': subject_bootstrap(
                y[test_indices],
                selected_test_prediction,
                test_groups,
            ),
            'candidate_comparison': final_results,
        },
        'models': {
            'existing': 'StandardScaler + MLPClassifier(48,24), x/y features',
            'improved_mlp': (
                'StandardScaler + regularized MLP candidates, x/y/z features, '
                'training-only class-balanced sample weights'
            ),
            'random_forest': (
                'RandomForestClassifier with balanced_subsample class weights, '
                'x/y/z features'
            ),
            'additional_baseline': (
                'RBF SVC with balanced class weights, x/y/z features; majority '
                'class dummy baseline included'
            ),
        },
        'feature_preprocessing': (
            'Per-frame deterministic shoulder-midpoint centering and division '
            'by 2-D shoulder width; StandardScaler is fit only on each training '
            'fold/split. XYZ is included where the candidate declares it.'
        ),
        'classes': list(CLASS_LABELS),
        'python': platform.python_version(),
        'scikit_learn': sklearn.__version__,
        'numpy': np.__version__,
        'accuracy_target_80_percent_achieved': (
            selected_entry['test']['accuracy'] >= 0.80
        ),
        'note': (
            'Probabilities are model scores, not calibrated clinical confidence. '
            'Offline participant-held-out results are not live webcam accuracy.'
        ),
    }

    selected_candidate = selected['candidate']
    model_features = list(
        FEATURE_NAMES_BY_DIMENSIONS[selected_candidate['dimensions']]
    )
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        'pipeline': selected_model,
        'features': model_features,
        'labels': list(CLASS_LABELS),
        'source': 'MultiPosture',
        'model_name': selected_name,
        'schema_version': 2,
        'feature_preprocessing': 'shoulder_midpoint_centered_2d_width_scaled',
    }, args.model_out, compress=3)
    args.metrics_out.write_text(
        json.dumps(metrics, indent=2, allow_nan=False),
        encoding='utf-8',
    )
    print(
        f"Selected by validation: {selected_name}; "
        f"test accuracy={selected_entry['test']['accuracy']:.3f}, "
        f"balanced accuracy={selected_entry['test']['balanced_accuracy']:.3f}, "
        f"macro-F1={selected_entry['test']['macro_f1']:.3f}; "
        f"80% target={'met' if metrics['accuracy_target_80_percent_achieved'] else 'not met'}"
    )
    print(f'Model saved to: {args.model_out}')
    print(f'Metrics saved to: {args.metrics_out}')


if __name__ == '__main__':
    main()
