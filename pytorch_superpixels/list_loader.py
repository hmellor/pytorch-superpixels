from os.path import exists, join


class ImageList:
    def __init__(self, dataset, path, split="trainval"):
        # Configured datasets
        datasets = {
            "pascal-seg": {
                "listPath": "ImageSets/Segmentation/",
                "imagePath": "JPEGImages",
                "targetPath": "SegmentationClass",
            }
        }
        # Object variables
        self.split = split
        self.dataset = dataset
        self.path = path
        self.listPath = join(path, datasets[dataset]["listPath"])
        self.imagePath = join(path, datasets[dataset]["imagePath"])
        self.targetPath = join(path, datasets[dataset]["targetPath"])
        self.list = []
        # Does the split exist?
        list_path = join(self.listPath, self.split + ".txt")
        if not exists(list_path):
            raise ValueError("The list you are looking for does not exist")
        # Open and parse the list
        self.list = tuple(open(list_path, "r"))
        self.list = [id_.rstrip() for id_ in self.list]
