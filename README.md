#  Library of pytoan

## Introduction

## Installing
```sh
pip install pytoan
```
## Usage
1. Setup Model
```python

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True,
                                           pin_memory=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False,
                                          pin_memory=True)

from pathlib import Path
def accuracy_score(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)
model = ConvNet(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
metric_ftns = [accuracy_score]
device = [0]
num_epoch = 20
gradient_clipping = 0.1
gradient_accumulation_steps = 1
early_stopping = 10
validation_frequency = 2
tensorboard = True
checkpoint_dir = Path('./', type(model).__name__)
checkpoint_dir.mkdir(exist_ok=True, parents=True)
resume_path = None
learning = Learning(model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler = scheduler,
                    metric_ftns=metric_ftns,
                    device=device,
                    num_epoch=num_epoch,
                    grad_clipping = gradient_clipping,
                    grad_accumulation_steps = gradient_accumulation_steps,
                    early_stopping = early_stopping,
                    validation_frequency = validation_frequency,
                    tensorboard = tensorboard,
                    checkpoint_dir = checkpoint_dir,
                    resume_path=resume_path)


```
2. For Training and Validation
```python
learning.train(train_loader, test_loader)
```
3. For Testing
```python
learning.test(test_loader) # but not complete
```