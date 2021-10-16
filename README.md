# PyTorch Superpixels
- [Why use superpixels?](#why-use-superpixels)
- [Example usage](#example-usage)
## Why use superpixels?
Dimensionality reduction allows for the use of simpler networks or more complex objectives. A common way of doing this is to simply downsample the images so that there are fewer pixels to contend with. However, this is a lossy operation so detail (and therefore the upper bound on experimental results) is reduced.

Superpixels slightly alleviate this problem because they are able to encode information about edges within themselves. Generating superpixels is an unsupervised clustering operation. Whilst there are already clustering packages written for Python (some of which this project depends on), they all operate with NumPy arrays. This means that they cannot take advantage of GPU acceleration in the way that PyTorch tensors can.

The aim of this project is to bridge the gap between these existing packages and PyTorch so that superpixels can be readily used as an alternative to pixels in various machine learning experiments.
## Example usage
Here is some example code that uses superpixels for semantic segmentation.
```
# Generate list of filenames from your dataset
imageList = pytorch_superpixels.list_loader.image_list(
    'pascal-seg', './VOCdevkit/VOC2012', 'trainval')
# Use this list to create and save 100 superpixel dataset
pytorch_superpixels.preprocess.create_masks(imageList, 100)

# -----------------------------------------------
# code that sets up model, optimizer, dataloader, metrics, etc.
# -----------------------------------------------

# Iterate through images, labels and masks
# (Note that masks and labels are already in superpixel form)
for (images, labels, masks) in trainloader:
        # Set model to training mode
        model.train()
        # Send images, labels and masks to the GPU
        images = images.to('cuda:0')
        labels = labels.to('cuda:0')
        masks = masks.to('cuda:0')
        # Pass images through the model
        optimizer.zero_grad()
        outputs = model(images)
        # Convert outputs to superpixel form
        outputs_sp, sizes = pixels_to_superpixels(
            outputs, masks)
        # Calculate loss using size weighted superpixels
        loss = loss_fn(scores=outputs_sp, target=labels, size=sizes)
        # Convert back to pixels for metric evaluation
        outputs = superpixels_to_pixels(outputs_sp, masks)
        # Accumulate train metrics during train
        pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels.data.cpu().numpy()
        running_metrics_train.update(gt, pred)
        train_loss_meter.update(loss.item())
        # Backprop and optimizer step
        loss.backward()
        optimizer.step()
```
_______________________________________

This project stems from a module I created for use in my master's thesis.

With some work I think it could be useful for others that wish to utilise superpixels in their Pytorch Machine learning projects.

Ideally I would like to add support for many of the more popular datasets, publish a pypi package and maybe even port this to TensorFlow.

Currently the code is fragmented and unusable, with community support I hope to change that soon.
