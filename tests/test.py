# Testing dataset generation
import pytorch_superpixels.preprocess
import pytorch_superpixels.list_loader


testList = pytorch_superpixels.list_loader.image_list(
    'pascal-seg', './VOCdevkit/VOC2012', 'trainval')
pytorch_superpixels.preprocess.create_masks(testList, 100)
