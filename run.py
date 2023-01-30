from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8x.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="data/10cv_ori/fold1.yaml", epochs=500, imgsz=512)