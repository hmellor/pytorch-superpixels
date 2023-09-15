from numpy.core.fromnumeric import product
from skimage.segmentation.boundaries import find_boundaries
import torch
import numpy as np
from torchvision.io import read_image
from torchvision.models.segmentation import fcn_resnet50
import matplotlib.pyplot as plt
from torchvision.transforms.functional import convert_image_dtype
from torchvision.utils import draw_segmentation_masks
from torchvision.utils import make_grid
from pytorch_superpixels.runtime import superpixelise
from skimage.segmentation import slic, mark_boundaries, find_boundaries
from pathlib import Path
from multiprocessing import Pool
from os import cpu_count
from functools import partial

import torchvision.transforms.functional as F

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sem_classes = [
        '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
    image_dims = [420, 640]
    images = [read_image(str(img)) for img in Path("data").glob("*.jpg")]
    images = [F.center_crop(image, image_dims) for image in images]
    image_size = product(image_dims)

    batch_int = torch.stack(images)
    batch = convert_image_dtype(batch_int, dtype=torch.float)

    # permute because slic expects the last dimension to be channel
    with Pool(processes = cpu_count()-1) as pool:
        # re-order axes for skimage
        args = [x.permute(1,2,0) for x in batch]
        # 100 segments
        kwargs = {"n_segments":100, "start_label":0, "slic_zero":True}
        func = partial(slic, **kwargs)
        masks_100sp = pool.map(func, args)
        # 1000 segments
        kwargs["n_segments"] = 1000
        func = partial(slic, **kwargs)
        masks_1000sp = pool.map(func, args)


    model = fcn_resnet50(pretrained=True, progress=False)
    model = model.eval()

    normalized_batch = F.normalize(batch, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    outputs = model(batch)['out']

    normalized_masks = torch.nn.functional.softmax(outputs, dim=1)
    num_classes = normalized_masks.shape[1]

    def generate_all_class_masks(outputs, masks):
        masks = np.stack(masks)
        masks = torch.from_numpy(masks)
        outputs_sp = superpixelise(outputs, masks)
        normalized_masks_sp = torch.nn.functional.softmax(outputs_sp, dim=1)
        return normalized_masks_sp[i].argmax(0) == torch.arange(num_classes)[:, None, None]

    to_show = []
    for i, image in enumerate(images):
        # before
        all_classes_masks = normalized_masks[i].argmax(0) == torch.arange(num_classes)[:, None, None]
        to_show.append(draw_segmentation_masks(image, masks=all_classes_masks, alpha=.6))
        # after 100
        all_classes_masks_sp = generate_all_class_masks(outputs, masks_100sp)
        to_show.append(draw_segmentation_masks(image, masks=all_classes_masks_sp, alpha=.6))
        # show superpixel boundaries
        boundaries = find_boundaries(masks_100sp[i])
        to_show[-1][0:2, boundaries] = 255
        to_show[-1][2, boundaries] = 0
        # after 1000
        all_classes_masks_sp = generate_all_class_masks(outputs, masks_1000sp)
        to_show.append(draw_segmentation_masks(image, masks=all_classes_masks_sp, alpha=.6))
        # show superpixel boundaries
        boundaries = find_boundaries(masks_1000sp[i])
        to_show[-1][0:2, boundaries] = 255
        to_show[-1][2, boundaries] = 0
    show(make_grid(to_show, nrow=6))
