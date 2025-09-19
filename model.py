import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, dropout, img_size, num_classes):
        super().__init__()
        self.dropout_rate = dropout
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            c = self.pool1(self.relu1(self.conv1(dummy_input)))
            c = self.pool2(self.relu2(self.conv2(c)))
            n_features = c.view(1, -1).size(1)

        self.fc1 = nn.Linear(n_features, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.activations = {}

    def forward(self, x):
        x = self.conv1(x); self.activations['conv1'] = x
        x = self.relu1(x); self.activations['relu1'] = x
        x = self.pool1(x); self.activations['pool1'] = x
        
        x = self.conv2(x); self.activations['conv2'] = x
        x = self.relu2(x); self.activations['relu2'] = x
        x = self.pool2(x); self.activations['pool2'] = x
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x); self.activations['fc1'] = x
        x = self.relu3(x); self.activations['relu3'] = x
        x = self.dropout(x); self.activations['dropout'] = x
        x = self.fc2(x); self.activations['fc2'] = x
        
        return x

