import argparse
from ultralytics import YOLO
import torch

def main():
    parser = argparse.ArgumentParser(description="Train YOLO on synthetic drone datasets.")
    parser.add_argument('--data', type=str, default='combined_dataset/data.yaml', help='Path to data.yaml file')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Base model to train (yolov8n.pt, yolo11n.pt, etc.)')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--batch', type=int, default=16, help='Batch size for training')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='0' if torch.cuda.is_available() else 'cpu', help='Device to run on (cpu, 0 for cuda:0, etc.)')
    args = parser.parse_args()

    print(f"Loading pretrained base model: {args.model}")
    model = YOLO(args.model)

    print(f"Starting training on device: {args.device} for {args.epochs} epochs...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project='drone_detection_challenge',
        name='yolov8_synthetic',
        plots=True  # Save training plots (results.png, confusion matrix etc.)
    )

    print("\nTraining completed successfully!")
    print("Check model weights and evaluation plots in: drone_detection_challenge/yolov8_synthetic/")

if __name__ == '__main__':
    main()
