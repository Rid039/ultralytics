from ultralytics import YOLO
# https://docs.ultralytics.com/cfg/

for i in range(1,2):
    # Load a model
    model = YOLO(f'/home/rid039/CB/TUNED/v8/ori/fold{i}/weights/best.pt')         # ori
    # model = YOLO(f'/home/rid039/CB/TUNED/v8/hnm_random/fold{i}/weights/best.pt')    # random
    # model = YOLO(f'/home/rid039/CB/TUNED/v8/hnm_overlap/fold{i}/weights/best.pt')    # overlap

    # Test the model
    model.val(data=f"data/10cv_ori/fold{i}.yaml")
    # model.val(data=f"data/10cv_predict_hard/fold{i}.yaml", save_json=True, conf=0.01)
    