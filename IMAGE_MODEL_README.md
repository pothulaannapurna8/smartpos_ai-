# SmartPos AI experimental image classifier

The original Flask dashboard, MediaPipe pose classifier, calibration, alerts and red overlay are retained. A new **Experimental image classifier** option is available under Detection Mode. Rule-based mode remains the default.

## Dataset and attribution

Roboflow *sitting posture* v4, https://universe.roboflow.com/test-rls1l/sitting-posture-flafm/dataset/4, CC BY 4.0. The image classifier was trained from 123 supplied JPG images (67 BAD, 56 GOOD). The original supplied validation partition has 21 BAD and zero GOOD, so it is not a valid two-class accuracy test.

## Training and evaluation

Run `python train_image_posture.py --dataset PATH_TO_EXTRACTED_YOLOV8_DATASET` to reproduce training. Training and webcam inference share the same grayscale conversion, 64x64 resize, histogram equalization, and HOG descriptor. The dashboard passes the original unmirrored camera frame to match the orientation used by `cv2.imread` during training; the displayed video is mirrored separately.

The reported grouped evaluation contains 44 test images (22 BAD, 22 GOOD): accuracy and balanced accuracy are both 86.36%; actual BAD/GOOD x predicted BAD/GOOD confusion matrix is `[[22, 0], [6, 16]]`. BAD precision/recall/F1 are 0.786/1.000/0.880; GOOD precision/recall/F1 are 1.000/0.727/0.842. The final model was then refitted on all 123 images. The small sample, same camera/subject/recording session and potentially similar frames mean these numbers **do not establish webcam or independent-person accuracy**. The original source images are not present in this project folder, so a separate near-duplicate audit could not be performed here. Timestamp grouping reduces adjacency leakage but cannot guarantee independent images or people. This is NOT an 80%-validated medical or production posture model.

## Run

Install dependencies from `requirements.txt` in a Python 3.12 virtual environment, then `python app.py` and open http://127.0.0.1:5000. Switch Detection Mode to **Experimental image classifier**. Keep head and shoulders visible. The image model is a GOOD/BAD classifier only; it cannot distinguish forward/backward/sideways leaning. It is independent from the five-class landmark model. Use Rule-based mode if image predictions are unstable. Calibration and neck-angle overlays are geometric approximations, not medical measurements.

## Limitations and further validation

The model's output probability is not calibrated confidence. The app requires a visible head and both shoulders before accepting image predictions to prevent an empty scene from producing a confident posture label. It reports GOOD/BAD only and cannot explain the posture issue or reliably distinguish forward, backward, or sideways lean. Evaluate on fresh videos of different people, cameras, clothing and lighting with independently labeled GOOD and BAD frames; preserve person/video groups across splits, inspect near duplicates, and report per-class precision/recall/F1 and a confusion matrix. If false positives persist, collect representative data from the target camera and users. Do not report the 86.36% grouped test as independent real-world accuracy. Preserve all original license/attribution notices when reusing project code.
