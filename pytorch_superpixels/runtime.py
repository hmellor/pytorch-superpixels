import torch
from os.path import join
from os.path import exists
from os.path import dirname
from os.path import abspath
from os import listdir

# Define absolute path for accessing dataset files
package_dir = dirname(abspath(__file__))
dataset_dir = "../../datasets/VOCdevkit/VOC2011"
root = join(package_dir, dataset_dir)
'''For use during runtime'''


def convert_to_superpixels(input, target, mask):
    # Extract size data from input and target
    images, c, h, w = input.size()
    if images > 1:
        raise RuntimeError("Not implemented for batch sizes greater than 1")
    # Initialise vairables to use
    Q = mask.unique().numel()
    output = torch.zeros((Q, c), device=input.device)
    size = torch.zeros(Q, device=input.device)
    counter = torch.ones(mask.size(), device=input.device)
    # Calculate the size of each superpixel
    size.put_(mask, counter, True)
    # Calculate the mean value of each superpixel
    input = input.view(c, -1)
    mask = mask.view(1, -1).repeat(c, 1)
    arange = torch.arange(start=1, end=c, device=input.device)
    mask[arange, :] += Q * arange.view(-1, 1)
    output = output.put_(mask, input, True).view(c, Q).t()
    output = (output.t() / size).t()
    return output, target.view(-1), size


def convert_to_pixels(input, output, mask):
    n, c, h, w = output.size()
    for k in range(c):
        output[0, k, :, :] = torch.gather(
            input[:, k], 0, mask.view(-1)).view(h, w)
    return output


def to_super_to_pixels(input, mask):
    target = torch.tensor([])
    input_s, _, _ = convert_to_superpixels(input, target, mask)
    output = convert_to_pixels(input_s, input, mask)
    return output


def setup_superpixels(superpixels):
    image_save_dir = join(
        root,
        "SegmentationClass/{}_sp".format(superpixels)
    )
    target_s_save_dir = join(
        root,
        "SegmentationClass/pre_encoded_{}_sp".format(superpixels)
    )
    dirs = [image_save_dir, target_s_save_dir]
    dataset_len = len(get_image_list())
    if not any(exists(x) and len(listdir(x)) == dataset_len for x in dirs):
            print("Superpixel dataset of scale {} superpixels either doesn't exist or is incomplete".format(superpixels))
            print("Generating superpixel dataset now...")
            create_masks(superpixels)

    fix_broken_images(superpixels)
