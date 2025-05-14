import torch
from torch.backends.mkl import verbose

from model import MyResnet
import torch.nn as nn
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


if __name__=="__main__":
    model = MyResnet()
    check_point = torch.load("check_point/best_model_f.pt", weights_only=True)
    model.load_state_dict(check_point["model"])
    categories = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
    model.eval()
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    fo = "image_test"
    for i, file_name in enumerate(os.listdir(fo)):
        if i >= 10:
            break
        path = os.path.join(fo, file_name)
        test = cv2.imread(path)
        test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB) / 255.
        test = cv2.resize(test, (32, 32))
        test = np.transpose(test, (2, 0, 1))
        test = test.reshape(1, 3, 32, 32).astype(np.float32)
        test = torch.from_numpy(test)
        output = model(test)
        soft_max = nn.Softmax(dim=1)
        vt = soft_max(output)
        idx = torch.argmax(vt, dim=1).item()
        probability = int(vt[0][idx].item() * 100)
        ax = axs[i // 5, i % 5]
        ax.imshow(test.squeeze().permute(1, 2, 0))
        ax.set_title(f"{categories[idx]}: {probability}%")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
