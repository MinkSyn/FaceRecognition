import os
import sys
from pathlib import Path

CWD = Path(__file__).resolve()
sys.path.append(CWD.parent.parent.as_posix())
sys.path.append(CWD.parent.parent.parent.as_posix())

import cv2
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import DataParallel
from sklearn.metrics import accuracy_score, recall_score

from const import CelebVNID, CONFIG_PATH, CONFIG_NAME
from arcface.model import resnet_face18


def verify_device(device):
    if device != 'cpu' and torch.cuda.is_available():
        return device
    return 'cpu'


def name_format(celeb):
    return celeb.replace(' ', '_')


class ArcFaceInference:
    def __init__(self, device, cfg):
        self.device = verify_device(device)
        self.img_size = cfg.img_size
        self.use_se = cfg.use_se
        self.model = self.load_model(weight_path=cfg.weight_path)

        self.threshold = cfg.threshold

        if os.path.exists(cfg.coresets_path):
            self.coresets = torch.load(cfg.coresets_path)
        else:
            self.coresets = None

    def load_model(self, weight_path):
        model = resnet_face18(use_se=self.use_se)
        model = DataParallel(model)
        model_state = torch.load(weight_path, map_location=self.device)

        try:
            model.load_state_dict(model_state)
        except:
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in model_state.items():
                name = k.replace('module.', '', 1)
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            
        model.to(self.device)
        model.eval()
        return model

    def predict(self, input, coresets=False):
        input = self._preprocess(input)

        output = self.model(input)

        output = output.detach().numpy()
        feature_1 = output[::2]
        feature_2 = output[1::2]
        feature = np.hstack((feature_1, feature_2))

        if coresets:
            return feature

        result = self._postprocess(feature)
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

    def _postprocess(self, feature: np.asarray):
        min_distance = 100
        for embedding in self.coresets:
            euclidean_distance = F.pairwise_distance(
                torch.from_numpy(feature), embedding['feature']
            )
            # euclidean_distance = self.cosin_metric(feature[0], embedding['feature'][0])
            if euclidean_distance < min_distance:
                id_pred = embedding['id']
                min_distance = euclidean_distance
        return id_pred, min_distance

    @staticmethod
    def cosin_metric(x1, x2):
        return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


@hydra.main(config_name=CONFIG_NAME, config_path=CONFIG_PATH, version_base=None)
def create_coresets(cfg):
    if os.path.exists(cfg.coresets.out_path):
        print("Coresets for embedding dataset already exists")
    else:
        print("Start creating coresets...")
        identify = ArcFaceInference(device=cfg.device, cfg=cfg.arcface)

        lst_celebs = sorted(os.listdir(cfg.coresets.root))

        embeddings_list = list()
        for celeb in lst_celebs:
            id_celeb = CelebVNID[name_format(celeb)].value
            celeb_path = os.path.join(cfg.coresets.root, celeb)

            for file_name in tqdm(
                sorted(os.listdir(celeb_path)), desc=f"Embdedding for [{celeb}]"
            ):
                file_path = os.path.join(celeb_path, file_name)
                img = cv2.imread(file_path)

                feature = identify.predict(img, coresets=True)

                embeddings_list.append({'feature': torch.from_numpy(feature), 'id': id_celeb})
        torch.save(embeddings_list, cfg.coresets.out_path)
        print(f"Finished creating coresets => out path: [{cfg.coresets.out_path}]")


@hydra.main(config_name=CONFIG_NAME, config_path=CONFIG_PATH, version_base=None)
def evaluate(cfg):
    identify = ArcFaceInference(device=cfg.device, cfg=cfg.arcface)

    lst_celebs = sorted(os.listdir(cfg.root_test))

    targets, predictions = [], []
    for celeb in lst_celebs:
        id_celeb = CelebVNID[name_format(celeb)].value

        celeb_path = os.path.join(cfg.root_test, celeb)

        for file_name in tqdm(
            sorted(os.listdir(celeb_path)), desc=f"Testing for [{celeb}]"
        ):
            file_path = os.path.join(celeb_path, file_name)
            img = cv2.imread(file_path)

            id_pred, _ = identify.predict(img)
            targets.append(id_celeb)
            predictions.append(id_pred)

    accuracy = accuracy_score(targets, predictions)
    recall = recall_score(targets, predictions, average='micro')
    print(f"Accuracy: {round(accuracy * 100, 2)}%") 
    print(f"Recall: {round(recall * 100, 2)}%")


@hydra.main(config_name=CONFIG_NAME, config_path=CONFIG_PATH, version_base=None)
def make_id(cfg):
    lst_celebs = sorted(os.listdir(cfg.coresets.root))

    for idx, celeb in enumerate(sorted(lst_celebs)):
        print(f"{name_format(celeb)} = {idx}")


@hydra.main(config_name=CONFIG_NAME, config_path=CONFIG_PATH, version_base=None)
def normalize_image_names(cfg):
    lst_roots = [cfg.coresets.root, cfg.root_test]

    for root_path in lst_roots:
        print(f"Processing for [{root_path}]")
        name_folder = root_path.split("/")[-1]
        new_folder = f"new_{name_folder}"
        new_root_path = '/'.join(root_path.split("/")[:-1])

        for celeb in sorted(os.listdir(root_path)):
            origin_path = os.path.join(root_path, celeb)
            new_path = os.path.join(new_root_path, new_folder, celeb)
            os.makedirs(new_path, exist_ok=True)

            for idx, file_name in tqdm(
                enumerate(sorted(os.listdir(origin_path))),
                desc=f"Formating image names for [{celeb}:]",
            ):
                origin_file = os.path.join(origin_path, file_name)
                new_file = os.path.join(new_path, f"{celeb}__{idx}.jpg")
                os.rename(origin_file, new_file)


if __name__ == '__main__':
    create_coresets()
    evaluate()

    # make_id()
    # normalize_image_names()
