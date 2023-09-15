from multiprocessing import cpu_count
from os import mkdir
from os.path import exists, join

import torch
from joblib import Parallel, delayed
from skimage.io import imread
from skimage.segmentation import slic
from skimage.util import img_as_float
from tqdm import tqdm


def create_masks(image_list, num_segments=100, oversegmentation_limit=None):
    # Save mask and target for image number
    def save_mask(image_number):
        # Load image/target pair
        image_path = join(image_list.imagePath, image_number + ".jpg")
        target_path = join(image_list.targetPath, image_number + ".png")
        image = img_as_float(imread(image_path))
        target = imread(target_path)
        target = torch.from_numpy(target)
        # Save paths
        save_dir = join(image_list.path, "SuperPixels")
        mask_dir = join(save_dir, "{}_sp_mask".format(num_segments))
        targetDir = join(save_dir, "{}_sp_target".format(num_segments))
        # Check that directories exist
        if not exists(save_dir):
            mkdir(save_dir)
        if not exists(mask_dir):
            mkdir(mask_dir)
        if not exists(targetDir):
            mkdir(targetDir)
        # Define save paths
        mask_save_path = join(mask_dir, image_number + ".pt")
        target_save_path = join(targetDir, image_number + ".pt")
        # If they haven't already been made, make them
        if not exists(mask_save_path) and not exists(target_save_path):
            # Create mask for image/target pair
            mask, target_s = create_mask(
                image=image,
                target=target,
                num_segments=num_segments,
                oversegmentation_limit=oversegmentation_limit,
            )
            torch.save(mask, mask_save_path)
            torch.save(target_s, target_save_path)

    num_cores = cpu_count()
    inputs = tqdm(image_list.list)
    # Iterate through all images utilising all CPU cores
    Parallel(n_jobs=num_cores)(
        delayed(save_mask)(image_number) for image_number in inputs
    )


def create_mask(image, target, num_segments, oversegmentation_limit):
    # Perform SLIC segmentation
    mask = slic(image, n_segments=num_segments, slic_zero=True)
    mask = torch.from_numpy(mask)

    if oversegmentation_limit is not None:
        # Oversegmentation step
        superpixels = mask.unique().numel()
        overseg = superpixels
        for superpixel in range(superpixels):
            overseg -= 1
            # Define mask for superpixel
            segment_mask = mask == superpixel
            # Classes in this superpixel
            classes = target[segment_mask].unique(sorted=True)
            # Check if superpixel is on target boundary
            on_boundary = classes.numel() > 1
            # If current superpixel is on a gt boundary
            if on_boundary:
                # Find how many of each class is in superpixel
                class_hist = torch.bincount(target[segment_mask])
                # Remove zero elements
                class_hist = class_hist[class_hist.nonzero()].float()
                # Find minority class in superpixel
                min_class = min(class_hist)
                # Is the minority class large enough for oversegmentation
                above_threshold = min_class > class_hist.sum() * oversegmentation_limit
                if above_threshold:
                    # Leaving one class in supperpixel be
                    for c in classes[1:]:
                        # Adding to the oversegmentation offset
                        overseg += 1
                        # Add offset to class c in the mask
                        mask[segment_mask] += (
                            target[segment_mask] == c
                        ).long() * overseg

    # (Re)define how many superpixels there are and create target_s
    superpixels = mask.unique().numel()
    target_s = torch.zeros(superpixels, dtype=torch.long)
    for superpixel in range(superpixels):
        # Define mask for superpixel
        segment_mask = mask == superpixel
        # Apply mask, the mode for majority class
        target_s[superpixel] = target[segment_mask].view(-1).mode()[0]
    return mask, target_s
