from typing import Union
import logging
import warnings
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

def visualise(title: str, image: Union[torch.tensor, np.ndarray], bboxes: torch.tensor, labels: torch.tensor):
    # If this is a pytorch tensor in the CHW space, we need to convert back to the numpy HWC space
    if type(image) is torch.Tensor:
        image = np.transpose(image.numpy(), (1, 2, 0))

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_title(f'{title} {image.shape[0]}x{image.shape[1]}')
    for idx, box in enumerate(bboxes):
        image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (220, 0, 0), 3)
        image = cv2.putText(image, str(int(labels[idx])), (int(box[0]) + 10, int(box[1]) + 20), 0, 0.6, (220, 0, 0), 2)
    ax.set_axis_off()
    ax.imshow(image)
    plt.show()