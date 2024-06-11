import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from medmnist import OCTMNIST  # python -m pip install git+https://github.com/MedMNIST/MedMNIST.git


def get_dataloaders(train_batchsize, test_batchsize=500, exclude_classes=None):
    """
    Get OCT MedMNIST dataset as a binary classification problem of class 3 (normal) vs classes 0, 1, 2.
    """

    # get the datasets
    train_dataset = OCTMNIST(split='train', transform=transforms.ToTensor())
    test_dataset = OCTMNIST(split='test', transform=transforms.ToTensor())
    train_imgs, train_labels = train_dataset.imgs, train_dataset.labels
    test_imgs, test_labels = test_dataset.imgs, test_dataset.labels

    # filter out excluded classes
    if exclude_classes is not None:
        for e in exclude_classes:
            train_imgs = train_imgs[(train_labels != e).squeeze()]
            train_labels = train_labels[(train_labels != e).squeeze()]
            test_imgs = test_imgs[(test_labels != e).squeeze()]
            test_labels = test_labels[(test_labels != e).squeeze()]

    # convert to a binary classification problem
    train_labels = (train_labels != 3)  # i.e. 0 = normal, 1 = abnormal
    test_labels = (test_labels != 3)

    # apply the appropriate scaling and transposition
    train_imgs = torch.tensor(train_imgs, dtype=torch.float32).unsqueeze(1) / 255
    test_imgs = torch.tensor(test_imgs, dtype=torch.float32).unsqueeze(1) / 255
    train_labels = torch.tensor(train_labels, dtype=torch.int64)
    test_labels = torch.tensor(test_labels, dtype=torch.int64)

    # form dataloaders
    train_dataset = TensorDataset(train_imgs, train_labels)
    test_dataset = TensorDataset(test_imgs, test_labels)
    dl = DataLoader(dataset=train_dataset, batch_size=train_batchsize, shuffle=True)
    dl_test = DataLoader(dataset=test_dataset, batch_size=test_batchsize, shuffle=True)
    return dl, dl_test