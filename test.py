# Testing dataset generation
import ptsupix.preprocess
import ptsupix.list_loader


testList = ptsupix.list_loader.image_list(
    'pascal-seg', './VOCdevkit/VOC2012', 'trainval')
ptsupix.preprocess.create_masks(testList, 100)
