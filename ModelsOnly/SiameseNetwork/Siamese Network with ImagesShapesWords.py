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

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(SiameseNetwork, self).__init__()
        # Separate CNNs for feature extraction from images and shapes
        self.image_cnn = ImageCNN()
        self.shape_cnn = ShapeCNN()
        # Additional fully connected layers for combining features
        self.fc_combined = nn.Linear(128 * 4 + embedding_dim, 128)  # Input size includes output size of CNNs and embedding size
        
    def forward(self, image_input1, shape_input1, image_input2, shape_input2, semantic_input):
        image_output1 = self.image_cnn(image_input1)
        shape_output1 = self.shape_cnn(shape_input1)
        image_output2 = self.image_cnn(image_input2)
        shape_output2 = self.shape_cnn(shape_input2)
        # Concatenate the feature vectors
        combined_output = torch.cat((image_output1, shape_output1, image_output2, shape_output2, semantic_input), 1)
        # Pass through the additional fully connected layer for combining features
        combined_output = F.relu(self.fc_combined(combined_output))
        return combined_output

class ClassificationNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ClassificationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate Siamese network
siamese_net = SiameseNetwork(embedding_dim)

# Instantiate classification networks
classification_net_1 = ClassificationNetwork(128 + embedding_dim, 1)  # For shape match vs connection
classification_net_2 = ClassificationNetwork(128 + embedding_dim, num_classes)  # For type of connection (e.g., fügen, stecken, schrauben)

# Define loss function and optimizer
criterion_1 = nn.BCEWithLogitsLoss()
criterion_2 = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(siamese_net.parameters()) + list(classification_net_1.parameters()) + list(classification_net_2.parameters()), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for data in train_loader:  # Assuming train_loader is defined
        image_inputs, shape_inputs, semantic_inputs, labels = data
        image_input1, image_input2 = image_inputs
        shape_input1, shape_input2 = shape_inputs
        semantic_input = semantic_inputs
        
        # Forward pass through Siamese network
        combined_output = siamese_net(image_input1, shape_input1, image_input2, shape_input2, semantic_input)
        
        # Forward pass through classification networks
        output_classification_1 = classification_net_1(combined_output)
        output_classification_2 = classification_net_2(combined_output)
        
        # Compute loss
        loss1 = criterion_1(output_classification_1, labels_classification_1)
        loss2 = criterion_2(output_classification_2, labels_classification_2)
        total_loss = loss1 + loss2
        
        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
