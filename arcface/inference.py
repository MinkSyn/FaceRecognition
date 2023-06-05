import cv2
import numpy as np
import torch

from arcface.model import resnet_face18


def verify_device(device):
    if device != 'cpu' and torch.cuda.is_available():
        return device
    return 'cpu'


class ArcFaceInference:
    def __init__(self, device, cfg):        
        self.device = verify_device(device)
        self.img_size = cfg.img_size
        self.use_se = cfg.use_se
        self.model = self.load_model(weight_path=cfg.weight_path)
        
        self.threshold = cfg.threshold
        
        # self.coresets = torch.load(cfg.embedding_path)
        
    def load_model(self, weight_path):
        model = resnet_face18(use_se=self.use_se)
        model_state = torch.load(weight_path, map_location=self.device)

        try:
            model.load_state_dict(model_state)
        except: 
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in model_state.items():
                name = k.replace('module.', '', 1)
                # name = k.replace('model.', '', 1)
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            
        return model.to(self.device)
    
    def predict(self, input):
        input = self._preprocess(input)

        output = self.model(input)

        result = self._postprocess(output)
        return result

    def _preprocess(self, input):
        if isinstance(input, str):  # path
            input = cv2.imread(input, 0)
        elif isinstance(input, np.ndarray):
            input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
            
        img_resize = cv2.resize(input, (self.img_size))
        img_lst = np.dstack((img_resize, np.fliplr(img_resize)))
        img_lst = img_lst.transpose((2, 0, 1))
        img_lst = img_lst[:, np.newaxis, :, :]
        image_nor = img_lst.astype(np.float32, copy=False)
        
        image_nor -= 127.5
        image_nor /= 127.5
        
        img_tensor = torch.from_numpy(image_nor)
        img_tensor = img_tensor.to(self.device)
        return img_tensor

    def _postprocess(self, output):
        output = output.detach().numpy()
        feature_1 = output[::2]
        feature_2 = output[1::2]
        feature = np.hstack((feature_1, feature_2))
        
        #TODO: embedding and return result
        
        return feature
