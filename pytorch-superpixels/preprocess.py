from skimage.io import imread
from skimage.segmentation import slic
from skimage.util import img_as_float
from os.path import exists
from os.path import join
from tqdm import tqdm
from os import mkdir
import torch


def create_masks(imageList, numSegments=100, limOverseg=None):
    # Iterate through all images
    for image_number in tqdm(imageList.list):
        # Load image/target pair
        image_path = join(imageList.imagePath, image_number + ".jpg")
        target_path = join(imageList.targetPath, image_number + ".png")
        image = img_as_float(imread(image_path))
        target = imread(target_path)
        target = torch.from_numpy(target)
        # Save paths
        saveDir = join(imageList.path, 'SuperPixels')
        maskDir = join(saveDir, '{}_sp_mask'.format(numSegments))
        targetDir = join(saveDir, '{}_sp_target'.format(numSegments))
        # Check that directories exist
        if not exists(saveDir):
            mkdir(saveDir)
        if not exists(maskDir):
            mkdir(maskDir)
        if not exists(targetDir):
            mkdir(targetDir)
        # Define save paths
        mask_save_path = join(maskDir, image_number + ".pt")
        target_save_path = join(targetDir, image_number + ".pt")
        # If they haven't already been made, make them
        if not exists(mask_save_path) and not exists(target_save_path):
            # Create mask for image/target pair
            mask, target_s = create_mask(
                image=image,
                target=target,
                numSegments=numSegments,
                limOverseg=limOverseg
            )
            torch.save(mask, mask_save_path)
            torch.save(target_s, target_save_path)


def create_mask(image, target, numSegments, limOverseg):
    # Perform SLIC segmentation
    mask = slic(image, n_segments=numSegments, slic_zero=True)
    mask = torch.from_numpy(mask)

    if limOverseg is not None:
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
                above_threshold = min_class > class_hist.sum() * limOverseg
                if above_threshold:
                    # Leaving one class in supperpixel be
                    for c in classes[1:]:
                        # Adding to the oversegmentation offset
                        overseg += 1
                        # Add offset to class c in the mask
                        mask[segment_mask] += (target[segment_mask]
                                               == c).long() * overseg

    # (Re)define how many superpixels there are and create target_s
    superpixels = mask.unique().numel()
    target_s = torch.zeros(superpixels, dtype=torch.long)
    for superpixel in range(superpixels):
        # Define mask for superpixel
        segment_mask = mask == superpixel
        # Apply mask, the mode for majority class
        target_s[superpixel] = target[segment_mask].view(-1).mode()[0]
    return mask, target_s
