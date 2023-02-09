import numpy as np
from ultralytics import YOLO
# https://docs.ultralytics.com/cfg/

# for hnm in ['ori', 'hnm_random', 'hnm_overlap']:
#     for conf in np.linspace(0.01, 1, 100):
#         for i in range(1,11):
#             # Load a model
#             # model = YOLO(f'/home/rid039/CB/TUNED/v8/ori/fold{i}/weights/best.pt')         # ori
#             # model = YOLO(f'/home/rid039/CB/TUNED/v8/hnm_random/fold{i}/weights/best.pt')    # random
#             # model = YOLO(f'/home/rid039/CB/TUNED/v8/hnm_overlap/fold{i}/weights/best.pt')    # overlap
#             model = YOLO(f'/home/rid039/CB/TUNED/v8/{hnm}/fold{i}/weights/best.pt')

#             # Test the model
#             # model.val(data=f"data/10cv_predict_hard/fold{i}.yaml", save_json=True, conf=0.01)
#             model.val(data=f"data/10cv_ori/fold{i}.yaml", report_mode=3, weight_path=model.ckpt_path, conf=conf)



for hnm in ['ori']:
    for conf in np.linspace(0.01, 1, 100):
        for i in range(1,11):
            # Load a model
            # model = YOLO(f'/home/rid039/CB/TUNED/v8/ori/fold{i}/weights/best.pt')         # ori
            # model = YOLO(f'/home/rid039/CB/TUNED/v8/hnm_random/fold{i}/weights/best.pt')    # random
            # model = YOLO(f'/home/rid039/CB/TUNED/v8/hnm_overlap/fold{i}/weights/best.pt')    # overlap
            model = YOLO(f'/home/rid039/CB/TUNED/v8/{hnm}/fold{i}/weights/best.pt')

            # Test the model
            # model.val(data=f"data/10cv_predict_hard/fold{i}.yaml", save_json=True, conf=0.01)
            # model.val(data=f"data/10cv_ori/fold{i}.yaml", report_mode=3, weight_path=model.ckpt_path, conf=conf)
            model.val(data=f"data/negative_for_ori.yaml", report_mode=4, weight_path=model.ckpt_path, conf=conf)

for hnm in ['hnm_random', 'hnm_overlap']:
    for conf in np.linspace(0.01, 1, 100):
        for i in range(1,11):
            # Load a model
            # model = YOLO(f'/home/rid039/CB/TUNED/v8/ori/fold{i}/weights/best.pt')         # ori
            # model = YOLO(f'/home/rid039/CB/TUNED/v8/hnm_random/fold{i}/weights/best.pt')    # random
            # model = YOLO(f'/home/rid039/CB/TUNED/v8/hnm_overlap/fold{i}/weights/best.pt')    # overlap
            model = YOLO(f'/home/rid039/CB/TUNED/v8/{hnm}/fold{i}/weights/best.pt')

            # Test the model
            # model.val(data=f"data/10cv_predict_hard/fold{i}.yaml", save_json=True, conf=0.01)
            # model.val(data=f"data/10cv_ori/fold{i}.yaml", report_mode=3, weight_path=model.ckpt_path, conf=conf)
            model.val(data=f"data/negative.yaml", report_mode=4, weight_path=model.ckpt_path, conf=conf)
            