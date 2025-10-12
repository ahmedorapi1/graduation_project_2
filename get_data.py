from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from custom_model import Model



class Data:
    def __init__(self, root, batch_size):
        self.root = root
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_data(self):
        dataset = datasets.ImageFolder(root=self.root, transform=self.transform)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        num_classes = len(dataset.classes)

        print(f"Loaded {len(dataset)} images from '{self.root}'")
        print(f"Detected {num_classes} classes: {dataset.classes}")

        return loader, num_classes
