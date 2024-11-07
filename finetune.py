import argparse
from ultralytics import YOLO
import torch

# Load a model
# model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
# model = YOLO('yolo11s.pt')  # load a custom model

# # Train the model
# results = model.train(data="./dataset/soda/soda.yaml", epochs=100, imgsz=640, batch=32)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./dataset/soda/soda.yaml', help='dataset.yaml path')
    parser.add_argument('--model', type=str, default='yolo11n', help='initial model.pt path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device(f'cuda:{args.device}')
    model = YOLO(f'{args.model}.pt')
    results = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=device)
    print(results)

if __name__ == '__main__':
    main()

# cmd: 
    # python finetune.py --data ./dataset/soda/soda.yaml --model yolo11s --epochs 100 --batch 32 --imgsz 640 --device 0
    # python finetune.py --data ./dataset/soda/soda.yaml --model yolo11n --epochs 100 --batch 32 --imgsz 640 --device 1
