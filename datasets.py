import torchvision
from utils.general_utils import *

class CIFAR10(torchvision.datasets.CIFAR10):
    """A subclass to suppress an annoying print statement in the torchvision CIFAR-10 library.

    Not strictly necessary - you can just use `torchvision.datasets.CIFAR10 if the print
    message doesn't bother you.
    """

    def download(self):
        super(CIFAR10, self).download() #Nope
        # with open(os.devnull, 'w') as fp:
        #     sys.stdout = fp
        #     super(CIFAR10, self).download()
        #     sys.stdout = sys.


class CIFAR10Dataset(ImageDataset):
    """The CIFAR-10 dataset."""

    @staticmethod
    def num_train_examples(): return 50000

    @staticmethod
    def num_test_examples(): return 10000

    @staticmethod
    def num_classes(): return 10

    @staticmethod
    def get_train_set(use_augmentation):
        augment = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(32, 4)]
        train_set = CIFAR10(train=True, root=os.path.join(os.path.join(pathlib.Path.home(), 'open_lth_datasets'), 'cifar10'), download=True)
        return CIFAR10Dataset(train_set.data, np.array(train_set.targets), augment if use_augmentation else [])

    @staticmethod
    def get_test_set():
        test_set = CIFAR10(train=False, root=os.path.join(os.path.join(pathlib.Path.home(), 'open_lth_datasets'), 'cifar10'), download=True)
        return CIFAR10Dataset(test_set.data, np.array(test_set.targets))

    def __init__(self,  examples, labels, image_transforms=None):
        super(CIFAR10Dataset, self).__init__(examples, labels, image_transforms or [],
                                      [torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def example_to_image(self, example):
        return Image.fromarray(example)


class CIFAR100Dataset(ImageDataset):
    """The CIFAR-10 dataset."""

    @staticmethod
    def num_train_examples(): return 50000

    @staticmethod
    def num_test_examples(): return 10000

    @staticmethod
    def num_classes(): return 100

    @staticmethod
    def get_train_set(use_augmentation):
        augment = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(32, 4)]
        train_set = torchvision.datasets.CIFAR100(train=True,
                                                  root=os.path.join(os.path.join(pathlib.Path.home(),
                                                                                 'open_lth_datasets'), 'cifar10'),
                                                  download=True)
        return CIFAR100Dataset(train_set.data, np.array(train_set.targets), augment if use_augmentation else [])

    @staticmethod
    def get_test_set():
        test_set = torchvision.datasets.CIFAR100(train=False,
                                                 root=os.path.join(os.path.join(pathlib.Path.home(),
                                                                                'open_lth_datasets'),
                                                                   'cifar10'), download=True)
        return CIFAR100Dataset(test_set.data, np.array(test_set.targets))

    def __init__(self,  examples, labels, image_transforms=None):
        super(CIFAR100Dataset, self).__init__(examples, labels, image_transforms or [],
                                      [torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def example_to_image(self, example):
        return Image.fromarray(example)


class CIFAR100CoarseDataset(ImageDataset):
    """The CIFAR-10 dataset."""
    # https://discuss.pytorch.org/t/cifar-100-targets-labels-doubt/81323/2
    label_map = {
        0: [72, 4, 95, 30, 55],
        1: [73, 32, 67, 91, 1],
        2: [92, 70, 82, 54, 62],
        3: [16, 61, 9, 10, 28],
        4: [51, 0, 53, 57, 83],
        5: [40, 39, 22, 87, 86],
        6: [20, 25, 94, 84, 5],
        7: [14, 24, 6, 7, 18],
        8: [43, 97, 42, 3, 88],
        9: [37, 17, 76, 12, 68],
        10: [49, 33, 71, 23, 60],
        11: [15, 21, 19, 31, 38],
        12: [75, 63, 66, 64, 34],
        13: [77, 26, 45, 99, 79],
        14: [11, 2, 35, 46, 98],
        15: [29, 93, 27, 78, 44],
        16: [65, 50, 74, 36, 80],
        17: [56, 52, 47, 59, 96],
        18: [8, 58, 90, 13, 48],
        19: [81, 69, 41, 89, 85],
    }

    fine_to_coarse_label = {}
    for coarse_label in label_map:
        for fine_label in label_map[coarse_label]:
            fine_to_coarse_label[fine_label] = coarse_label

    def __getitem__(self, index):
        example, fine_label = super(CIFAR100CoarseDataset, self).__getitem__(index)
        return example, CIFAR100CoarseDataset.fine_to_coarse_label[fine_label]

    @staticmethod
    def num_train_examples(): return 50000

    @staticmethod
    def num_test_examples(): return 10000

    @staticmethod
    def num_classes(): return 20

    @staticmethod
    def get_train_set(use_augmentation):
        augment = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(32, 4)]
        train_set = torchvision.datasets.CIFAR100(train=True,
                                                  root=os.path.join(os.path.join(pathlib.Path.home(),
                                                                                 'open_lth_datasets'), 'cifar10'),
                                                  download=True)
        for fine_label in range(100):
            train_set.targets[train_set.targets == fine_label] = CIFAR100CoarseDataset.fine_to_coarse_label[fine_label]
        return CIFAR100CoarseDataset(train_set.data, np.array(train_set.targets), augment if use_augmentation else [])

    @staticmethod
    def get_test_set():
        test_set = torchvision.datasets.CIFAR100(train=False,
                                                 root=os.path.join(os.path.join(pathlib.Path.home(),
                                                                                'open_lth_datasets'),
                                                                   'cifar10'), download=True)
        for fine_label in range(100):
            test_set.targets[test_set.targets == fine_label] = CIFAR100CoarseDataset.fine_to_coarse_label[fine_label]
        return CIFAR100CoarseDataset(test_set.data, np.array(test_set.targets))

    def __init__(self,  examples, labels, image_transforms=None):
        super(CIFAR100CoarseDataset, self).__init__(examples, labels, image_transforms or [],
                                      [torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def example_to_image(self, example):
        return Image.fromarray(example)



cifar10 = argparse.Namespace(
    CIFAR10=CIFAR10,
    Dataset=CIFAR10Dataset,
    DataLoader=DataLoader,
)

cifar100 = argparse.Namespace(
    CIFAR100=torchvision.datasets.CIFAR100,
    Dataset=CIFAR100Dataset,
    DataLoader=DataLoader,
)

cifar100_coarse = argparse.Namespace(
    CIFAR100=torchvision.datasets.CIFAR100,
    Dataset=CIFAR100CoarseDataset,
    DataLoader=DataLoader,
)

# registered_datasets = {'cifar10': cifar10, 'mnist': mnist, 'imagenet': imagenet}
registered_datasets = {'cifar10': cifar10, 'cifar100': cifar100, 'cifar100_coarse': cifar100_coarse}