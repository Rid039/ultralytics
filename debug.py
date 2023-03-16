import numpy as np
from PIL import Image
from ultralytics import YOLO
# https://docs.ultralytics.com/cfg/


# hnm = 'hnm_overlap'
hnm = 'morph'
i = 1
model = YOLO(f'/home/rid039/CB/TUNED/v8/8x6/scratch_{hnm}/fold{i}/weights/best.pt') # Load a model


# Test the model
# model.val(data=f"data/10cv_predict_hard/fold{i}.yaml", save_json=True, conf=0.01)
result = model('/home/rid039/CB/datasets/cb_1203_cleaned/10cv_ori/fold1/images/test/S16006330_A3_1_177959_116792.jpg')
print(result)

res_plotted = result[0].plot()
im = Image.fromarray(res_plotted)
im.save("zzz.jpeg")