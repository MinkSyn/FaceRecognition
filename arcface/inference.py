from typing import Union

import numpy as np
import torch
from PIL import Image

from model import resnet_face18
from tool import is_image_file, verify_device, get_tfms, to_device


class ArcFaceInference:
    def __init__(self, device='cuda', img_size=(128, 128), weight_path=None):
        self.device = verify_device(device)
        model = resnet_face18()
        ckpt = torch.load(weight_path)

        model.load_state_dict(ckpt)
        self.model = model.to(self.device)
        self.tfms = get_tfms(
            img_size=img_size, norm_stats=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        )

    def _preprocess(self, input):
        if isinstance(input, np.ndarray):
            img = input[:, :, ::-1].copy()
        elif isinstance(input, str):  # path
            assert is_image_file(input), f"{input} is not an image."
            with open(input, 'rb') as f:
                img = np.array(Image.open(f).convert('RGB'))
            assert img is not None, f"{input} is not a valid path."
        else:
            raise NotImplementedError(f"{type(input)} is not supported.")

        img = self.tfms(img)
        img = img.unsqueeze(0)
        img = to_device(img, self.device)
        return img

    def predict(self, input):
        input = self._preprocess(input)

        output = self.model(input)

        result = self._postprocess(output)

        return result

    def _postprocess(self, input):
        return input
