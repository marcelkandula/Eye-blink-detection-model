import os
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

class_map = {
    'closedLeftEyes': 0,
    'closedRightEyes': 0,
    'openLeftEyes': 1,
    'openRightEyes': 1
}


class EyeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # We load whole dataset from subfolders and put label on each image
        self.samples = []
        self.transform = transform
        for folder, label in class_map.items():
            folder_path = os.path.join(root_dir, folder)
            for fname in os.listdir(folder_path):
                if fname.endswith('.jpg'):
                    self.samples.append((os.path.join(folder_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = default_loader(path)
        if self.transform:
            image = self.transform(image)
        return image, label