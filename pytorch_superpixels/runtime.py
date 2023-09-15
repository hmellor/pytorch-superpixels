import torch

"""For use during runtime"""


def pixels_to_superpixels(inputs, masks):
    n, c, h, w = inputs.size()
    img_size = h * w
    # Initialise vairables to use
    sizes = [torch.unique(x, return_counts=True)[1] for x in masks]
    num_superpixels = [size.numel() for size in sizes]
    outputs = [torch.zeros((Q, c)) for Q in num_superpixels]
    # reshape tensors
    inputs = inputs.view(n, c, img_size)
    masks_copy = masks.view(n, 1, img_size).repeat(1, c, 1)
    arange = torch.arange(start=0, end=c).view(c, 1).expand(-1, img_size)
    for i in range(n):
        # Calculate the mean value of each superpixel
        masks_copy[i] += num_superpixels[i] * arange
        outputs[i] = (
            outputs[i]
            .put_(masks_copy[i], inputs[i], True)
            .view(c, num_superpixels[i])
            .t()
        )
        outputs[i] = (outputs[i].t() / sizes[i]).t()
    return outputs, sizes


def superpixels_to_pixels(superpixel_images, pixel_images, masks):
    n, c, h, w = pixel_images.size()
    superpixelised_images = torch.zeros_like(pixel_images)
    for i in range(n):
        for k in range(c):
            superpixelised_images[i, k, :, :] = torch.gather(
                superpixel_images[i][:, k], 0, masks[i].view(-1)
            ).view(h, w)
    return superpixelised_images


def superpixelise(pixel_images, masks):
    superpixel_images, _ = pixels_to_superpixels(pixel_images, masks)
    superpixelised_images = superpixels_to_pixels(
        superpixel_images, pixel_images, masks
    )
    return superpixelised_images
