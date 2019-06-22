'''For pre-processing'''


def create_masks(numSegments=100, limOverseg=None):
    # Generate image list
    image_list = get_image_list()
    for image_number in tqdm(image_list):
        # Load image/target pair
        image_name = image_number + ".jpg"
        target_name = image_number + ".png"
        image_path = join(root, "JPEGImages", image_name)
        target_path = join(root, "SegmentationClass/pre_encoded", target_name)
        image = img_as_float(io.imread(image_path))
        target = io.imread(target_path)
        target = torch.from_numpy(target)
        # Create mask for image/target pair
        mask, target_s = create_mask(
            image=image,
            target=target,
            numSegments=numSegments,
            limOverseg=limOverseg
        )

        # Save for later
        image_save_dir = join(
            root,
            "SegmentationClass/{}_sp".format(numSegments)
        )
        target_s_save_dir = join(
            root,
            "SegmentationClass/pre_encoded_{}_sp".format(numSegments)
        )
        if not exists(image_save_dir):
            mkdir(image_save_dir)
        if not exists(target_s_save_dir):
            mkdir(target_s_save_dir)
        save_name = image_number + ".pt"
        image_save_path = join(image_save_dir, save_name)
        target_s_save_path = join(target_s_save_dir, save_name)
        torch.save(mask, image_save_path)
        torch.save(target_s, target_s_save_path)


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


def get_image_list(split=None):
    if split is None:
        image_list_path = join(root, "ImageSets/Segmentation/trainval.txt")
    else:
        image_list_path = join(root, "ImageSets/Segmentation/", split + ".txt")
    image_list = tuple(open(image_list_path, "r"))
    image_list = [id_.rstrip() for id_ in image_list]
    return image_list
