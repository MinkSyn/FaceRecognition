import os

import cv2
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score

from const import CelebVNID, CONFIG_PATH, CONFIG_NAME
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

        if cfg.get('embedding_path', None) is None:
            self.coresets = None
        else:
            self.coresets = torch.load(cfg.embedding_path)

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

    def predict(self, input, coresets=False, eval=False):
        input = self._preprocess(input)

        output = self.model(input)
        
        output = output.detach().numpy()
        feature_1 = output[::2]
        feature_2 = output[1::2]
        feature = np.hstack((feature_1, feature_2))
        
        if coresets:
            return feature

        result = self._postprocess(feature, eval)
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

    def _postprocess(self, feature):
        min_distance = 100
        for embedding in self.coresets:
            euclidean_distance = F.pairwise_distance(feature, embedding['feature'])
            if euclidean_distance < min_distance:
                id_pred = embedding['id']
        return id_pred


@hydra.main(config_name=CONFIG_NAME, config_path=CONFIG_PATH, version_base=None)
def create_coresets(cfg):
    identify = ArcFaceInference(device=cfg.device, cfg=cfg.arcface)

    lst_celebs = sorted(os.listdir(cfg.root_coresets))

    embeddings_list = list()
    for celeb in lst_celebs:
        id_celeb = CelebVNID[celeb].values()

        celeb_path = os.path.join(cfg.root_test, celeb)

        for file_name in sorted(os.listdir(celeb_path)):
            file_path = os.path.join(celeb_path, file_name)
            img = cv2.imread(file_path)

            feature = identify.predict(img, coresets=True)

            embeddings_list.append({'feature': feature, 'id': id_celeb})
    torch.save(embeddings_list, cfg.coresets_path)


@hydra.main(config_name=CONFIG_NAME, config_path=CONFIG_PATH, version_base=None)
def main(cfg):
    identify = ArcFaceInference(device=cfg.device, cfg=cfg.arcface)

    lst_celebs = sorted(os.listdir(cfg.root_test))

    targets, predictions = [], []
    for celeb in lst_celebs:
        id_celeb = CelebVNID[celeb].values()

        celeb_path = os.path.join(cfg.root_test, celeb)

        for file_name in sorted(os.listdir(celeb_path)):
            file_path = os.path.join(celeb_path, file_name)
            img = cv2.imread(file_path)

            id_pred = identify.predict(img)

            targets.append(id_celeb)
            predictions.append(id_pred)

    accuracy = accuracy_score(targets, predictions, normalize=False)
    recall = recall_score(targets, predictions)
    print(f"Accuracy: {accuracy}, Recall: {recall}")


if __name__ == '__main__':
    main()
