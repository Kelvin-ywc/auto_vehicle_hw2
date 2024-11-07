# training for yolo11s
python finetune.py --data ./dataset/soda/soda.yaml --model yolo11s --epochs 100 --batch 32 --imgsz 640 
# training for yolo11n
python finetune.py --data ./dataset/soda/soda.yaml --model yolo11n --epochs 100 --batch 32 --imgsz 640 