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
    parser.add_argument('--img_path', type=str, default='./assets/images.jpg')
    parser.add_argument('--model', type=str, default='runs/detect/yolo11n_e100/weights/best.pt', help='model weights path')
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--conf', type=float, default=0.5, help='object confidence threshold')
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device(f'cuda:{args.device}')
    model = YOLO(f'{args.model}')

    # Run inference on 'bus.jpg' with arguments
    results = model.predict(args.img_path, save=True, imgsz=args.imgsz, conf=args.conf, device=device)
    print(results)

if __name__ == '__main__':
    main()