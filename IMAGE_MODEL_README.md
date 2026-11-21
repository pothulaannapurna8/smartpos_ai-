# Binary GOOD/BAD image classifier

The existing `image_posture_model.joblib` is an experimental HOG + logistic-regression classifier. It predicts only GOOD or BAD and cannot identify a lean direction. The app uses it as independent diagnostic evidence in default hybrid mode. Because its live webcam performance is not yet validated, an AI BAD conflicting with geometric GOOD produces UNKNOWN (not BAD); image-only mode is available for testing.

## Dataset and attribution

Roboflow *sitting posture* v4: https://universe.roboflow.com/test-rls1l/sitting-posture-flafm/dataset/4, licensed CC BY 4.0. The image classifier was trained from 123 supplied JPG images (67 BAD, 56 GOOD). The original supplied validation partition has 21 BAD and zero GOOD, so that partition is not a valid two-class accuracy test. Preserve this attribution and license information when redistributing the model or training code.

## Training and preprocessing

Training and inference call the same `image_features()` implementation in `smartpos/image_features.py`:

1. Accept a three-channel BGR image (`cv2.imread` format or original OpenCV webcam frame).
2. Convert BGR to grayscale.
3. Resize the **full frame** to 64×64 with `INTER_AREA`; no crop is applied.
4. Apply histogram equalization.
5. Extract the 1,764-element OpenCV HOG descriptor.

The camera worker supplies the original, unmirrored BGR frame to the image model, matching the training loader's BGR `cv2.imread` path. The displayed webcam is mirrored separately for the user. The saved model bundle declares `hog_image_binary_v1`, labels `['BAD', 'GOOD']`, class indices `[0, 1]`, and 1,764 features; inference checks this schema before use. `0` maps to BAD and `1` maps to GOOD. The prediction probability is uncalibrated; low-confidence predictions remain UNKNOWN while preserving raw label/confidence in diagnostics.

To reproduce training with an authorized extracted YOLOv8 dataset:

```powershell
.\.venv\Scripts\python.exe train_image_posture.py --dataset PATH_TO_EXTRACTED_DATASET
```

Training overwrites the specified model output after evaluating a grouped same-session split. Do not retrain from webcam validation sessions until their baseline metrics and failure causes have been reviewed.

## Existing offline evaluation

The saved report used 44 test images (22 BAD, 22 GOOD): accuracy and balanced accuracy were both 86.36%; actual BAD/GOOD × predicted BAD/GOOD confusion matrix `[[22, 0], [6, 16]]`. BAD precision/recall/F1 were 0.786/1.000/0.880; GOOD precision/recall/F1 were 1.000/0.727/0.842. The final model was subsequently refit on all 123 images.

The sample is small and grouped within the supplied data; it does **not** establish independent-person, webcam, or medical accuracy. The source images are not present in this project folder, so a separate near-duplicate audit was not possible. Timestamp grouping reduces adjacency leakage but cannot guarantee independent images or people. Do not claim 80%+ live accuracy without separate measurement.

## Live validation

Use the local consent-based capture and evaluation utility documented in [`README.md`](README.md#consent-based-webcam-validation). It stores samples under the ignored `webcam-validation/` directory, uses distinct calibration/evaluation sessions, reports UNKNOWN outcomes explicitly, and never trains a model. New users, cameras, lighting, clothing, crop/layout and camera viewpoints can cause domain shift even though the preprocessing code matches. No webcam validation is claimed by this document.
