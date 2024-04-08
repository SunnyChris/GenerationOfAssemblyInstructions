import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        return x

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Separate CNN for feature extraction
        self.cnn = CNN()
        # Additional fully connected layer for binary classification
        self.fc_binary = nn.Linear(128 * 2, 1)  # Input size is twice the output size of CNN
        
    def forward(self, input1, input2):
        output1 = self.cnn(input1)
        output2 = self.cnn(input2)
        # Concatenate the feature vectors
        combined_output = torch.cat((output1, output2), 1)
        # Pass through the additional fully connected layer for binary classification
        binary_output = torch.sigmoid(self.fc_binary(combined_output))
        return output1, output2, binary_output

class ClassificationNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ClassificationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size * 2, 64)  # Combine input sizes of both images
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate Siamese network
siamese_net = SiameseNetwork()

# Instantiate classification networks
classification_net_1 = ClassificationNetwork(128, 1)  # For shape match vs connection
classification_net_2 = ClassificationNetwork(128, num_classes)  # For type of connection (e.g., fügen, stecken, schrauben)

# Define loss function and optimizer
criterion_1 = nn.BCEWithLogitsLoss()
criterion_2 = nn.CrossEntropyLoss()
criterion_3 = nn.BCELoss()  # Binary classification loss
optimizer = torch.optim.Adam(list(siamese_net.parameters()) + list(classification_net_1.parameters()) + list(classification_net_2.parameters()), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for data in train_loader:  # Assuming train_loader is defined
        inputs, labels, binary_labels = data
        input1, input2 = inputs
        
        # Forward pass durch Siamese network
        output1, output2, binary_output = siamese_net(input1, input2)
        
        # Concatenate the feature vectors
        combined_output = torch.cat((output1, output2), 1)
        
        # Forward pass durch classification networks
        output_classification_1 = classification_net_1(combined_output)
        output_classification_2 = classification_net_2(combined_output)
        
        # Compute loss
        loss1 = criterion_1(output_classification_1, labels_classification_1)
        loss2 = criterion_2(output_classification_2, labels_classification_2)
        loss3 = criterion_3(binary_output, binary_labels)  # Binary classification loss
        total_loss = loss1 + loss2 + loss3  # Sum of all losses
        
        # Backward pass und optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
