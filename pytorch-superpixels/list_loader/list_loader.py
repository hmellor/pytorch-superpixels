from os.path import join


class image_list:
    def __init__(self, dataset, path, split='trainval'):
        datasets = ['pascal']
        splits = [None, 'train', 'val', 'trainval']

        datasets = {'pascal': {'listPath': 'ImageSets/Segmentation/',
                               'imagePath': 'JPEGImages',
                               'targetPath': 'SegmentationClass'}
                    }

        if dataset in datasets and split in splits:
            self.split = split
            self.dataset = dataset
            self.path = path
            self.listPath = join(path, datasets[dataset]['listPath'])
            self.imagePath = join(path, datasets[dataset]['imagePath'])
            self.targetPath = join(path, datasets[dataset]['targetPath'])
            self.list = []
        else:
            raise ValueError("Invalid dataset and/or split")

        list_path = join(self.listPath, self.split + ".txt")
        self.list = tuple(open(list_path, "r"))
        self.list = [id_.rstrip() for id_ in self.list]
