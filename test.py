from pytoan.pytorch import Learning

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm 

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

from pathlib import Path
def accuracy_score(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)
  
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
device = [0]
num_epoch = 20
gradient_clipping = 0.1
gradient_accumulation_steps = 1
early_stopping = 5
validation_frequency = 1
saved_period = 1
checkpoint_dir = Path('./', type(model).__name__)
checkpoint_dir.mkdir(exist_ok=True, parents=True)
resume_path = None
metric_ftns = [accuracy_score]
learning = Learning(model=model,
                        optimizer=optimizer,
                        criterion=criterion,
                        device=device,
                        metric_ftns=metric_ftns,
                        num_epoch=num_epoch,
                        scheduler = scheduler,
                        grad_clipping = gradient_clipping,
                        grad_accumulation_steps = gradient_accumulation_steps,
                        early_stopping = early_stopping,
                        validation_frequency = validation_frequency,
                        save_period = saved_period,
                        checkpoint_dir = checkpoint_dir,
                        resume_path=resume_path)


learning.train(tqdm(train_loader), tqdm(test_loader))