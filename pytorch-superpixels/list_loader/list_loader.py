from os.path import join


class image_list:
    def __init__(self, dataset, path, split=None):
        datasets = {
            'pascal': "ImageSets/Segmentation/",
        }
        splits = [None, 'train', 'val', 'trainval']
        if dataset in datasets and split in splits:
            self.split = split
            self.dataset = dataset
            self.path = join(path, datasets[dataset])
            self.list = []
        else:
            raise ValueError("Invalid dataset and/or split")

        if split is None:
            list_path = join(self.path, "trainval.txt")
        else:
            list_path = join(self.path, split + ".txt")
        self.list = tuple(open(list_path, "r"))
        self.list = [id_.rstrip() for id_ in self.list]
