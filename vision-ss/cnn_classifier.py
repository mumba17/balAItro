import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.data['Label'].unique())}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, str(self.data.iloc[idx, 0])).join(['', '.png'])
        try:
            image = Image.open(img_name).convert('L')  # Convert to grayscale
        except Exception as e:
            print(f"Error loading image {img_name}: {str(e)}")
            image = Image.new('L', (128, 64), color=0)  # Black grayscale image

        label = self.class_to_idx[self.data.iloc[idx, 1]]

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Error applying transform to image {img_name}: {str(e)}")
                image = torch.zeros((1, 128, 64))  # Single channel for grayscale

        return image, label

class CNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # 1 input channel for grayscale
        self.conv2 = nn.Conv2d(32, 90, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # Calculate the size of the feature maps after convolutions and pooling
        self.feature_size = self._get_conv_output((1, 160, 90))
        
        self.fc1 = nn.Linear(self.feature_size, 160)
        self.fc2 = nn.Linear(160, num_classes)

    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.1f}%)\n')

def main():
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((160, 90)),
        transforms.ToTensor(),
    ])

    # Create custom dataset
    dataset = CustomImageDataset(csv_file='template_labels.csv', img_dir='templates', transform=transform)

    # Split the dataset into train and test sets
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = CNN(num_classes=len(dataset.class_to_idx)).to(device)

    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and test
    for epoch in range(1, 11):  # 10 epochs
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    # Save the model
    torch.save(model.state_dict(), 'models/cnn_classifier.pth')
    print("Training completed. Model saved as 'cnn_classifier.pth'")

if __name__ == '__main__':
    main()