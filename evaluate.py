import argparse
from ultralytics import YOLO
import torch

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model and explain metrics.")
    parser.add_argument('--weights', type=str, default='runs/detect/drone_detection_challenge/yolov8_synthetic/weights/best.pt', help='Path to trained YOLO weights (best.pt)')
    parser.add_argument('--data', type=str, default='combined_dataset/data.yaml', help='Path to data.yaml file to validate against')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='0' if torch.cuda.is_available() else 'cpu', help='Device to run on (cpu, cuda:0, etc.)')
    args = parser.parse_args()

    print(f"Loading trained weights: {args.weights}")
    try:
        model = YOLO(args.weights)
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Please make sure you have run the training script successfully first and that the best.pt file exists.")
        return

    print(f"Running evaluation on {args.data} using device {args.device}...")
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        split='val'
    )

    print("\n================ EVALUATION METRICS SUMMARY ================")

    # Extract metrics directly from the metrics object (works across all Ultralytics versions)
    try:
        precision = metrics.box.mp        # Mean Precision across all classes
        recall    = metrics.box.mr        # Mean Recall across all classes
        map50     = metrics.box.map50     # mAP at IoU threshold 0.50
        map50_95  = metrics.box.map       # mAP at IoU thresholds 0.50-0.95
    except AttributeError:
        # Fallback to results_dict for older Ultralytics versions
        precision = metrics.results_dict.get('metrics/precision(B)', 0.0)
        recall    = metrics.results_dict.get('metrics/recall(B)', 0.0)
        map50     = metrics.results_dict.get('metrics/mAP50(B)', 0.0)
        map50_95  = metrics.results_dict.get('metrics/mAP50-95(B)', 0.0)

    print(f"Precision:   {precision:.4f} (Ability to avoid false positives)")
    print(f"Recall:      {recall:.4f} (Ability to detect all real drones)")
    print(f"mAP@0.50:    {map50:.4f} (Primary competition metric, Mean Average Precision at IoU threshold 0.5)")
    print(f"mAP@50-95:   {map50_95:.4f} (Average Precision across IoU thresholds from 0.5 to 0.95)")
    print("============================================================\n")

    print("Detailed explanation of metrics:")
    print("1. Precision (P): Out of all drone detections the model made, what percentage were actually drones? High precision means fewer false alarms (like mistaking a bird for a drone).")
    print("2. Recall (R): Out of all actual drones present in the images, what percentage did the model detect? High recall means the model rarely misses a drone.")
    print("3. mAP@0.50: Mean Average Precision at Intersection over Union (IoU) threshold of 0.50. This measures how well the model places the bounding boxes. If a box overlaps at least 50% with the real drone and class matches, it is counted as correct. This is the primary scoring metric.")
    print("4. mAP@50-95: A stricter metric that averages mAP over multiple IoU thresholds (from 0.50 up to 0.95 in steps of 0.05). It shows how tight and accurate the bounding boxes are.")

if __name__ == '__main__':
    main()
