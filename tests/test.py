# Testing dataset generation
import pytorch_superpixels.list_loader
import pytorch_superpixels.preprocess

test_list = pytorch_superpixels.list_loader.ImageList(
    "pascal-seg", "./VOCdevkit/VOC2012", "trainval"
)
pytorch_superpixels.preprocess.create_masks(test_list, 100)
