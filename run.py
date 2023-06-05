import cv2
import hydra
import torch

from yolov7.experimental import attempt_load
from yolov7.utils.general import check_img_size

from arcface.inference import ArcFaceInference

class Inference:
    def __init__(self, img_size, device, weight_path):
        self.yolo_model, self.stride, self.img_size = self.get_yolo_model(
            weight_path, device, img_size
        )

    def get_yolo_model(weight_path, device, img_size):
        model = attempt_load(weight_path, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(img_size, s=stride)  # check img_size
        
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        
        return model, stride, imgsz

    def run(self, img_path):
        input = self.preprocsess_yolo(img_path)

        detect_box = self.yolo_model(input)

    def preprocsess_yolo(self, input):
        # TODO
        return input


@hydra.main(config_name='config', config_path='E:/projects/src', version_base=None)
def main(cfg):
    img_path = 'E:/projects/data/images/Ai Phuong/AI_PHUONG.jpg'
    img_np = cv2.imread(img_path)
    
    detect = None
    identify = ArcFaceInference(device=cfg.device, cfg=cfg.arcface)
    
    result = identify.predict(img_path)
    print(result.shape)


if __name__ == '__main__':
    main()
