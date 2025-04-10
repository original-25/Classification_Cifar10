import matplotlib.pyplot as plt
import numpy as np
def plot_confusion_matrix(writer, cm, class_names, epoch):
    cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm, cmap="summer")

    # Hiển thị giá trị trên từng ô
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    writer.add_figure("Confusion Matrix", fig, epoch)