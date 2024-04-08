import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageCNN(nn.Module):
    def __init__(self):
        super(ImageCNN, self).__init__()
        # Convolutional layers for feature extraction from images
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        return x

class ShapeCNN(nn.Module):
    def __init__(self):
        super(ShapeCNN, self).__init__()
        # Convolutional layers for feature extraction from shapes
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        return x

class WordMLP(nn.Module):
    def __init__(self, embedding_dim):
        super(WordMLP, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class CombinedMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CombinedMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate CNNs and MLPs
image_cnn = ImageCNN()
shape_cnn = ShapeCNN()
word_mlp = WordMLP(embedding_dim)
combined_mlp = CombinedMLP(128 + 128 + 128, num_classes)  # Input size includes output sizes of CNNs and MLP

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(image_cnn.parameters()) + list(shape_cnn.parameters()) + list(word_mlp.parameters()) + list(combined_mlp.parameters()), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for data in train_loader:  # Assuming train_loader is defined
        images, shapes, words, labels = data
        
        # Forward pass through CNNs and MLPs
        image_output = image_cnn(images)
        shape_output = shape_cnn(shapes)
        word_output = word_mlp(words)
        
        # Concatenate the outputs
        combined_output = torch.cat((image_output, shape_output, word_output), 1)
        
        # Forward pass through combined MLP
        output = combined_mlp(combined_output)
        
        # Compute loss
        loss = criterion(output, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
