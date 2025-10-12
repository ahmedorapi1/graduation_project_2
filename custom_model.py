import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model:
    def __init__(self, data_batch, num_classes, num_epochs):
        self.num_classes = num_classes
        self.data_batch = data_batch
        self.num_epochs = num_epochs
        self.main_model = 'resnet50_custom_2.pth'
        self.model = self.get_model().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def get_model(self):
        resnet = models.resnet50(weights=None)
        resnet.fc = nn.Linear(2048, 36)

        try:
            state_dict = torch.load(self.main_model, map_location=device)
            resnet.load_state_dict(state_dict)
            print("✅ Pretrained weights loaded successfully.")
        except FileNotFoundError:
            print("⚠️ Warning: Model weights not found. Training from scratch.")

        for param in resnet.parameters():
            param.requires_grad = False

        resnet.fc = nn.Linear(2048, self.num_classes)
        return resnet

    def train_batch(self, x, y):
        self.model.train()
        preds = self.model(x)
        loss = self.criterion(preds, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def fine_tune(self):
        for epoch in range(self.num_epochs):
            for x, y in self.data_batch:
                x, y = x.to(device), y.to(device)
                self.train_batch(x, y)
        return self.model


    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in self.data_batch:
                x, y = x.to(device), y.to(device)
                preds = self.model(x)
                _, predicted = torch.max(preds, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
        return 100 * correct / total

