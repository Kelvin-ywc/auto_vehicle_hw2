import argparse
from numpy import save
from ultralytics import YOLO
import torch



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', type=str, default='./dataset/soda/soda.yaml', help='dataset.yaml path')
    parser.add_argument('--model_path', type=str, default='./runs/detect/yolo11n_e100/weights/best.pt', help='initial model.pt path')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    # parser.add_argument('--save_dir', type=str, default='./runs/val_yolo11n', help='save directory')
    return parser.parse_args()

def main():
    args = get_args()
    model = YOLO(f"{args.model_path}")
    results = model.val(data=f"{args.data_config}", imgsz=640)
    print(results)  # Print mAP50-95
if __name__ == '__main__':
    main()