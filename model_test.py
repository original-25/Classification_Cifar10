import torch
from torch.backends.mkl import verbose

from model import MyResnet
import torch.nn as nn
import cv2
import numpy as np
import os


if __name__=="__main__":
    model = MyResnet()
    check_point = torch.load("check_point/best_model_f.pt", weights_only=True)
    print(check_point["accuracy"])
    model.load_state_dict(check_point["model"])

    model.eval()

    fo = "image_test"
    for file_name in os.listdir(fo):
        path = os.path.join(fo, file_name)
        test = cv2.imread(path)
        test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)/255
        test = cv2.resize(test, (32, 32))
        test = np.transpose(test, (2, 0, 1))
        test = test.reshape(1, 3, 32, 32).astype(np.float32)
        test = torch.from_numpy(test)
        output = model(test)
        soft_max = nn.Softmax(dim=1)
        vt = soft_max(output)
        categories = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
        idx = torch.argmax(vt, dim=1).item()
        print(file_name, categories[idx], int(vt[0][idx].item()*100))
