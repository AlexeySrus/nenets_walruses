import cv2
import torch
import numpy as np
import os


class YoungWalrusesClassier(object):
    def __init__(self, model_path: str = None, conf: float=0.8):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '../data/young_model.pt')

        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.conf = conf

    def __call__(self, image: np.ndarray) -> int:
        cv2.imwrite('/home/alexey/Downloads/TEST_img.jpg', image)
        inp_tensor = torch.FloatTensor(
            cv2.cvtColor(cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        ).permute(2, 0, 1).unsqueeze(0) / 255.0

        out = torch.softmax(self.model(inp_tensor).detach(), dim=1)
        cls_res = out[0].to('cpu').numpy().argmax()
        return int(out[0].to('cpu').numpy()[1] > self.conf)
