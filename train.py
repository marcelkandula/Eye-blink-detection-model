from model import EyeBlinkCNN
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import EyeDataset
DATASET_DIR = 'dataset'
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 0.0005
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4)),  
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = EyeDataset(DATASET_DIR, transform=train_transforms)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = EyeBlinkCNN().to(DEVICE)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

'''
    Standard training loop for a classification using cross-entropy loss and the Adam optimizer.
    For each epoch, the model is trained on all batches and accuracy is calculated over the full training set.

'''
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100 * correct / total
    print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')


torch.save(model.state_dict(), 'eye_blink_cnn_BrightnessAugment.pth')
