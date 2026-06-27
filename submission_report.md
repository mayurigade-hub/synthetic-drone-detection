# ECSoC 2026 Drone Detection Challenge — Phase 1: EO Submission Report

**Participant**: Mayuri (Elite Coders Summer of Code AI/ML Track)  
**Submission Modality**: Electro-Optical (EO) Drone Detection  
**Model Architecture**: YOLOv8 Nano (`yolov8n`)  
**GitHub Repository**: [github.com/mayurigade-hub/synthetic-drone-detection](https://github.com/mayurigade-hub/synthetic-drone-detection)  

---

## 1. Training Dataset Details
As per competition rules, **100% of the training data was generated synthetically using Duality AI Vibe Sim**. No real-world reference datasets were used for training.

* **Rural Dataset (`Hackathon_RuralCUAS`)**: 31 images, containing drone objects in randomized locations with active clutter objects (birds, manned aircraft) generated to mirror real-world object scale distributions.
* **Urban Dataset (`Hackathon_UrbanCUAS_ChangingWeather`)**: 29 images, featuring drone flights in city environments under randomized, changing weather conditions and clutter.
* **Dataset Links**:
  * **Rural Dataset Link**: [rural_dataset.zip (Google Drive)](https://drive.google.com/file/d/1Y5E7rK1hseMnQ-UlXNbatjFPHPoGA1r7/view?usp=sharing)
  * **Urban Dataset Link**: [urban_dataset.zip (Google Drive)](https://drive.google.com/file/d/1Gkqqw8f7KUJ7qrDo4oJ98iG_XrKwzq-A/view?usp=sharing)

---

## 2. Vibe Sim Generation Prompts
The following agent prompts were executed in Vibe Sim to achieve optimal synthetic data quality and mitigate the Sim-to-Real gap:

1. **Dataset Size**: `Set the number of dataset images to 200`
2. **False Positive Prevention (Clutter)**: `Enable birds` & `Enable manned aircrafts`
3. **Camera Pitch Range**: `Set the maximum altitude angle to 45.0 and minimum altitude angle to 10.0`
4. **Size Distribution Matching**: `Set the pixels on target distribution to Custom using ieee_pixels_on_target.npy` (mirrors real-world drone pixel sizes from the IEEE dataset).
5. **Weather Diversity (Urban)**: `Randomize weather conditions across the generated dataset`

---

## 3. Evaluation & Output Metrics
The YOLO model was trained for **15 epochs** with a batch size of **4** on the combined synthetic dataset. The model was validated on an independent split of 12 synthetic images (containing 36 total target annotations).

### Performance Metrics Table

| Class Name | Images | Instances | Precision (P) | Recall (R) | mAP@0.50 (Primary Metric) | mAP@0.50-0.95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Drone (Class 0)** | 12 | 12 | **0.6020** | **0.5000** | **0.4630** | **0.3850** |
| **Bird (Class 1)** | 12 | 12 | 0.7160 | 0.4170 | 0.4970 | 0.4160 |
| **Manned Aircraft (Class 2)** | 12 | 12 | 0.4490 | 0.4170 | 0.2720 | 0.1890 |
| **All Classes (Average)** | 12 | 36 | 0.5889 | 0.4444 | 0.4105 | 0.3296 |

---

## 4. Beginner's Guide to Metrics Explanation
For the final submission, here is a detailed breakdown of what these numbers mean:

* **Precision (0.6020 for Drone)**: 
  When the model identifies an object as a drone, it is correct **60.2%** of the time. The remaining 39.8% are false alarms (e.g., misclassifying a bird or aircraft as a drone).
* **Recall (0.5000 for Drone)**: 
  The model successfully detects **50.0%** of all actual drones present in the validation frames. It missed the other 50.0% (likely due to extremely small sizes of the drones or background blending).
* **mAP@0.50 (0.4630 for Drone)**: 
  Mean Average Precision evaluated at an Intersection-over-Union (IoU) threshold of 0.50. If the predicted bounding box overlaps by 50% or more with the actual drone's bounding box and correctly labels it, it is counted as a successful detection. This is the main evaluation score for the ECSoC challenge.
* **mAP@0.50-0.95 (0.3850 for Drone)**: 
  The average precision measured across multiple overlap thresholds (from 50% to 95%). A high score here indicates that the model is extremely precise at bounding box alignment and fits the drone tightly.
