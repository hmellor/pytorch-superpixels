'''For superpixel validation'''


def mask_accuracy(target, mask):
    target_s = torch.zeros_like(target)
    superpixels = mask.unique().numel()
    for superpixel in range(superpixels):
        # Define mask for cluster idx
        segment_mask = mask == superpixel
        # Take slices to select image, apply mask, mode for majority class
        target_s[segment_mask] = target[segment_mask].view(-1).mode()[0]
    accuracy = torch.mean((target == target_s).float())
    return accuracy


def dataset_accuracy(superpixels):
    # Generate image list
    if superpixels is not None:
        image_list = get_image_list('trainval_super')
    else:
        image_list = get_image_list()

    mask_acc = 0
    mask_dir = "SegmentationClass/{}_sp".format(superpixels)
    target_dir = "SegmentationClass/pre_encoded"
    for image_number in tqdm(image_list):
        mask_path = join(root, mask_dir, image_number + ".pt")
        target_path = join(root, target_dir, image_number + ".png")
        mask = torch.load(mask_path)
        target = io.imread(target_path)
        target = torch.from_numpy(target)
        mask_acc += mask_accuracy(target, mask)
    dataset_acc = mask_acc / len(image_list)
    return dataset_acc


def find_smallest_object():
    # Generate image list
    image_list = get_image_list()
    smallest_object = 1e6
    for image_number in tqdm(image_list):
        target_name = image_number + ".png"
        target_path = join(root, "SegmentationClass/pre_encoded", target_name)
        target = io.imread(target_path)
        target = torch.from_numpy(target)
        object_size = torch.ne(target, 0).sum()
        if object_size < smallest_object:
            smallest_object = object_size
            print(smallest_object, image_number)
    return smallest_object


def find_usable_images(split, superpixels):
    # Generate image list
    image_list = get_image_list(split)
    usable = []
    target_dir = join(
        root,
        "SegmentationClass/pre_encoded_{}_sp".format(superpixels)
    )
    for image_number in image_list:
        target_name = image_number + ".pt"
        target_path = join(target_dir, target_name)
        target = torch.load(target_path)
        if target.nonzero().numel() > 0:
            usable.append(image_number)
    return usable


def fix_broken_images(superpixels):
    for split in ["train", "val", "trainval"]:
        usable = find_usable_images(split=split, superpixels=superpixels)
        super_path = join(root, "ImageSets/Segmentation", split + "_super.txt")
        if exists(super_path):
            remove(super_path)
        with open(super_path, "w+") as file:
            for image_number in usable:
                file.write(image_number + "\n")


def find_size_variance(superpixels):
    # Generate image list
    if superpixels is not None:
        image_list = get_image_list('trainval_super')
    else:
        image_list = get_image_list()
    mask_dir = "SegmentationClass/{}_sp".format(superpixels)
    dataset_variance = 0
    for image_number in tqdm(image_list):
        mask_path = join(root, mask_dir, image_number + ".pt")
        mask = torch.load(mask_path)
        # Initialise number of superpixels tensors
        Q = mask.unique().numel()
        size = torch.zeros(Q)
        counter = torch.ones_like(mask)
        # Calculate the size of each superpixel
        size.put_(mask, counter.float(), True)
        # Calculate the mean and standard deviation of the sizes
        std = size.std()
        mean = size.mean()
        # Add to the variance of the total datasets
        dataset_variance += std / mean
    dataset_variance /= len(image_list)
    return dataset_variance
