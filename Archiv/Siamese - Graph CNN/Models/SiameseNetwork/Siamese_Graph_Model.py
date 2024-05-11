#!/usr/bin/env python
# coding: utf-8

# In[212]:


###Libraries
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torchvision.models as models

import os
import pandas as pd
from torchvision.io import read_image
import torchvision
import torchvision.transforms as T

from PIL import Image
from random import randint
import pandas as pd
import pickl
import pickle

from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

import copy


# In[254]:


###Arguments
img_dir_train = "GNNdataset2/train_resnet50"
img_dir_test = "GNNdataset2/test_resnet50"
annotations_file_train = "GNNdataset2/train_resnet50_edgelabels/labels.csv"
annotations_file_test = "GNNdataset2/test_resnet50_labels/labels.csv"
#img_dir_edges = "GNNdataset2/train_resnet50_edgelabels/edge_index_sparsely_connected.csv"
img_dir_eval_edges = "GNNdataset2/test_resnet50_labels/test_edge_label_index.csv"

number_of_samples = 1200
number_of_nodes_individual_graph = 8
model = models.resnet50(pretrained=True)

###TO BUILD FEATURES WITH GRAPH
train_graph_tensor = torch.load('GNNdataset2/train_resnet50_edgelabels/graph_tensor.pt')
train_feature_tensor = torch.load('GNNdataset2/train_resnet50_edgelabels/features_tensor.pt')
#custom_dataset = torch.load('GNNdataset2/train_resnet50_edgelabels/custom_dataset.pt')


# In[229]:


###LABELS: Fügen + Stecken + Schrauben
def create_left_number(initial_values, num_loops):
    # Liste für die aktualisierten Werte
    updated_values_list = []
    
    # Schleife für jeden Loop
    for loop in range(num_loops):
        # Jedes Element in der Liste um 8 erhöhen und zu den aktualisierten Werten hinzufügen
        updated_values_list.extend([value + loop * 8 for value in initial_values])

    # Liste in einen Tensor konvertieren
    updated_values_tensor = torch.tensor(updated_values_list)

    # Ausgabe des Tensors
    print("Tensor der aktualisierten Werte:")
    print(updated_values_tensor)
    return updated_values_tensor

# Beispielaufruf der Funktion
initial_values = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 6, 7]
num_loops = 150 #int((number_of_samples/number_of_nodes_individual_graph)-1)
left_numbers = create_left_number(initial_values, num_loops)


# In[230]:


###LABELS: Fügen + Stecken + Schrauben
def create_second_number_and_label(left_numbers, initial_values, num_loops):
    # Liste für die aktualisierten Werte
    updated_values_list = []
    right_number_count = 0
    delta = 24
    label = 0
    #graph_count = 0
    
    for count, value in enumerate(left_numbers):
        left_number_value = left_numbers[count] 
        right_number_value = initial_values[right_number_count]
        # Schleife für jeden Loop
        for loop in range(num_loops): ###Default loops +1
            # Jedes Element in der Liste um 8 erhöhen und zu den aktualisierten Werten hinzufügen
            pair = np.append(left_number_value, right_number_value)
            pair = pair.tolist()
            pair.append(label)
            print(pair)
            updated_values_list.append(pair)
            right_number_value = right_number_value + delta
        right_number_count = right_number_count + 1
        
        if right_number_count == 18:
            right_number_count = 0
            
        if right_number_count <= 13:
            delta = 24
            num_loops = 50
            label = 0
        else:
            delta = 8
            num_loops = 150
            if right_number_count <= 15:
                label = 1
            else:
                label = 2
                    
    # Liste in einen Tensor konvertieren
    updated_values_tensor = torch.tensor(updated_values_list)

    # Ausgabe des Tensors
    print("Tensor der aktualisierten Werte:")
    print(updated_values_tensor)
    return updated_values_tensor

# Beispielaufruf der Funktion
initial_values = [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 0, 1, 0, 1, 6, 5, 7, 6]
num_loops = 50 #int((number_of_samples/number_of_nodes_individual_graph)-1)
graph_data = create_second_number_and_label(left_numbers, initial_values, num_loops)


# In[231]:


import random
import torch

def add_negative_samples(positive_samples):
    negative_samples = []
    count = 0
    
    # Konvertiere positive Samples in eine Liste von Listen
    positive_samples_list = positive_samples.tolist()
    
    # Berechne die Anzahl der negativen Samples basierend auf einem Prozentsatz der positiven Samples
    num_negative_samples = int(0.4 * len(positive_samples))
    print("Number of negative samples to generate:", num_negative_samples)
    
    # Generiere negative Samples
    while len(negative_samples) < num_negative_samples:
        # Zufällig Indizes für die ersten und zweiten Bauteile generieren
        first_component = random.randint(0, 1199)  # Hier 1199 für 1200 mögliche Bauteile
        second_component = random.randint(0, 1199)
        
        count = count + 1
        
        # Überprüfen, ob das generierte Bauteilpaar bereits in den positiven Samples vorhanden ist
        found = False
        for label in range(3):  # Durchlaufe alle Labels (0, 1, 2)
            if [first_component, second_component, label] in positive_samples_list:
                found = True
                break
        
        # Füge das Bauteilpaar mit dem Label 4 den negativen Samples hinzu
        if not found:
            negative_samples.append([first_component, second_component, 3])
            print("Added negative sample:", [first_component, second_component, 3], "Loop: ", count)
    
    # Konvertiere die negative Samples in einen Tensor
    negative_samples_tensor = torch.tensor(negative_samples)
    
    return negative_samples_tensor

# Beispielaufruf der Funktion mit positiven Samples
# Hier positiven_samples durch den aktuellen Tensor ersetzen
negative_samples = add_negative_samples(graph_data)

# Ausgabe der negativen Samples
print("Negative Samples:")
print(negative_samples)


# In[232]:


print(len(graph_data))
print(len(negative_samples))


# In[233]:


# Hinzufügen des negativen Tensors zum aktuellen Tensor
print("Amount positive samples: ", len(graph_data))
print("Amount negative samples: ", len(negative_samples))
all_graph_data = torch.cat((graph_data, negative_samples), dim=0)
print("Amount all samples: ", len(all_graph_data))
print(all_graph_data[:10])
print(all_graph_data[-10:])


# In[234]:


###Hinzufügen der Binary Labels:
def add_fourth_element(tensor):
    # Eine leere Liste erstellen, um die modifizierten Zeilen zu speichern
    modified_rows = []
    
    # Über jede Zeile des Tensors iterieren
    for row in tensor:
        # Den Wert der dritten Stelle in der aktuellen Zeile abrufen
        third_element = row[2].item()
        
        # Die vierte Stelle basierend auf dem Wert der dritten Stelle festlegen
        fourth_element = 1 if third_element in [0, 1, 2] else 0
        
        # Die aktuelle Zeile um die vierte Stelle erweitern
        modified_row = torch.tensor([row[0].item(), row[1].item(), row[2].item(), fourth_element])
        
        # Die modifizierte Zeile zur Liste hinzufügen
        modified_rows.append(modified_row)
    
    # Den modifizierten Tensor erstellen, indem die Liste in einen Tensor umgewandelt wird
    modified_tensor = torch.stack(modified_rows)
    
    return modified_tensor

all_graph_data = add_fourth_element(all_graph_data)
print(all_graph_data[:10])
print(all_graph_data[-10:])


# In[235]:


# Zufälliges Mischen der aktuellen Daten
shuffled_data = all_graph_data.tolist()
random.shuffle(shuffled_data)
shuffled_data_tensor = torch.tensor(shuffled_data)
print(len(shuffled_data_tensor))
print(shuffled_data_tensor)

# Speichern des Tensors
torch.save(shuffled_data_tensor, 'GNNdataset2/train_resnet50_edgelabels/graph_tensor.pt')


# In[150]:


##### LOAD FEATURES OF IMAGES
def read_images(path):
    toys = []
    # creates a ScandirIterator aliased as files
    with os.scandir(path) as files:
        for file in files:
            if file.name.endswith('.jpg'):
                toys.append(path + '/' + file.name)
    return toys

def extract_features(file):
    backbone = nn.Sequential(*list(model.children())[:-1])
    backbone.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # load the image as a 224x224 array
    image = Image.open(file)
    image = transform(image)
    image = image.unsqueeze(0)
    out = backbone(image)
    # get the feature vector
    out = torch.mean(out, -1)
    out = torch.mean(out, -1)
    return out

def get_Resnet50_values(path):
    toys = read_images(path)
    data = {}
    for toy in toys:
        features = extract_features(toy)
        features = features.detach().numpy()
        data[toy] = features
    features = np.array(list(data.values()))
    features = features.reshape(-1,2048) #Default with Resnet value 2048
    print(features.shape)
    return features


###LOAD LABELS OF IMAGES
transform = T.Resize((64,64))
transform2 = T.Grayscale() #####

class CustomImageDataset():
    def __init__(self, annotations_file, img_dir, transform=torchvision.transforms.ConvertImageDtype(dtype=torch.float32), target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        #img = read_image(img_path)
        #img = transform(img) #####
        #img = transform2(img) ####
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        #return img, label
        return label
    
###LOAD GRAPH DATA OF EVAL_IMAGES

def get_edge_index_list(img_dir_edges):
    edges = []
    labels = []
    with open(img_dir_edges, "r") as f:
        for line in f:
            edge_data = list(map(int, line.strip().split(";")[0].split(",")))
            label_data = list(map(int, line.strip().split(";")[1].split(",")))
            #edges, label_data = map(int, line.strip().split(","))
            #edges.extend([edge_data, label_data])
            edges.append(edge_data)
            labels.append(label_data)
            
    img_edges = torch.tensor(edges)
    img_labels = torch.tensor(labels)
    eval_graph_data = torch.cat((img_edges, img_labels), dim=1)
    return eval_graph_data


# In[12]:


###GET FEATURES:
features = get_Resnet50_values(img_dir_train)
torch.save(features, 'GNNdataset2/train_resnet50_edgelabels/features_tensor.pt')


# In[13]:


###GET LABELS:
labels = CustomImageDataset(annotations_file_train, img_dir_train)
print(labels)


# In[14]:


###GET GRAPH:
#img_edges = get_edge_index_list(img_dir_edges)
#print(img_edges)


# In[248]:


### Combine Features mit Graph Information

import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, image_features_tensor, graph_info_tensor):
        self.image_features = image_features_tensor
        self.graph_info = graph_info_tensor

    def __len__(self):
        return len(self.graph_info)

    def __getitem__(self, idx):
        img_idx1, img_idx2, label, binary_labels = self.graph_info[idx]
        image1 = self.image_features[img_idx1]
        image2 = self.image_features[img_idx2]
        return image1, image2, label, binary_labels

# Beispiel-Tensoren für Bildfeatures und Graph-Informationen
image_features_tensor = torch.randn(1200, 2048)  # Beispiel: Zufällige Werte für 1200 Bilder und 2048 Merkmale
graph_info_tensor = torch.tensor([[0, 1, 0], [0, 2, 1], [1, 2, 2]])  # Beispiel: Drei Einträge für den Graphen

# Erstellen des benutzerdefinierten Datasets
#custom_dataset = CustomDataset(features, shuffled_data_tensor)

# Beispielaufruf des Datasets
#for i in range(len(custom_dataset)):
#    image1, image2, label = custom_dataset[i]
#    print(f"Datensatz {i}:")
#    print("Bild 1:", image1)
#    print("Bild 2:", image2)
#    print("Label:", label)
#    print()


# In[255]:


custom_dataset = CustomDataset(train_feature_tensor, train_graph_tensor)
torch.save(custom_dataset, 'GNNdataset2/train_resnet50_edgelabels/custom_dataset.pt')


# In[250]:


print(len(custom_dataset))


# In[238]:


###LOAD EVAL DATA:
features_eval = get_Resnet50_values(img_dir_test)


# In[301]:


eval_graph = get_edge_index_list(img_dir_eval_edges)
print(eval_graph)


# In[302]:


eval_graph = add_fourth_element(eval_graph)


# In[303]:


eval_dataset = CustomDataset(features_eval, eval_graph)


# In[304]:


eval_loader = DataLoader(eval_dataset, batch_size=24, shuffle=False)


# In[305]:


def transfer_predicted_labels_to_graph(eval_labels, eval_graph):
    # Eine tiefe Kopie von eval_labels erstellen
    eval_labels_copy = copy.deepcopy(eval_labels)
    eval_graph_copy = copy.deepcopy(eval_graph)

    # Die Liste von Listen in eine einzelne Liste umwandeln
    eval_labels_flat = [item for sublist in eval_labels_copy for item in sublist]

    # Die Predictions ersetzen die Werte in der dritten Spalte
    eval_graph_copy[:, 2] = torch.tensor(eval_labels_flat)
    return eval_graph_copy


def show_images_from_tensor(tensor, path):
    fig, axes = plt.subplots(24, 2, figsize=(10, 40))
    
    for i, (image_idx1, image_idx2, label, binary_label) in enumerate(tensor[:24]):
        image_path1 = os.path.join(path, f"{image_idx1:04}.jpg")
        image_path2 = os.path.join(path, f"{image_idx2:04}.jpg")

        img1 = Image.open(image_path1)
        img2 = Image.open(image_path2)

        axes[i, 0].imshow(img1)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(img2)
        axes[i, 1].axis('off')

        # Anzeige des Labels rechts neben den Bildern
        axes[i, 1].text(1.05, 0.5, f"Label: {label}",
                        verticalalignment='center', horizontalalignment='left',
                        transform=axes[i, 1].transAxes, fontsize=12)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0, hspace=0.5)
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


####LOAD ALL DATA AND START HERE


# In[346]:


def build_batches(custom_dataset, batch_size, data_size):
    subset = [custom_dataset[i] for i in range(data_size)]
    data_size2 = data_size + 800
    data_size3 = data_size2 + 800
    val_subset = [custom_dataset[i] for i in range(data_size, data_size2)]
    test_subset = [custom_dataset[i] for i in range(data_size2, data_size3)]
    print(len(subset))
    print(subset)
    train_loader = DataLoader(subset, batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=True)
    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = build_batches(custom_dataset, batch_size=120, data_size=1200)


# In[347]:


print(len(train_loader))
print(len(val_loader))
print(len(test_loader))


# In[348]:


###MODEL
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers for feature extraction
        #self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        #self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        #self.pool = nn.MaxPool2d(2, 2)
        #self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),   # Reduziere die Dimension auf 512
            nn.ReLU(),
            nn.Linear(1024, 512),     # Reduziere die Dimension auf 64
            nn.ReLU())
        
    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = torch.flatten(x, 1) # flatten all dimensions except batch
        #x = F.relu(self.fc1(x))
        x = self.fc(x)
        return x

class SiameseNetwork(nn.Module):
    def __init__(self, num_classes):
        super(SiameseNetwork, self).__init__()
        # Separate CNN for feature extraction
        self.cnn = CNN()
        self.fc1 = nn.Linear(512 * 2, 64)  # Combine input sizes of both images
        self.fc2 = nn.Linear(64, num_classes)
        # Additional fully connected layer for binary classification
        self.fc_binary = nn.Linear(512 * 2, 1)  # Input size is twice the output size of CNN
        
    def forward(self, input1, input2):
        output1 = self.cnn(input1)
        output2 = self.cnn(input2)
        # Concatenate the feature vectors
        combined_output = torch.cat((output1, output2), 1)
        x = F.relu(self.fc1(combined_output))
        x = self.fc2(x)
        # Pass through the additional fully connected layer for binary classification
        binary_output = torch.sigmoid(self.fc_binary(combined_output))
        return x, binary_output #output1, output2#


# In[349]:


###INITIATE MODEL:
# Instantiate Siamese network
siamese_net = SiameseNetwork(num_classes = 4)

# Define loss function and optimizer
criterion_1 = nn.CrossEntropyLoss()
criterion_2 = nn.BCELoss()  # Binary classification loss
#optimizer = torch.optim.Adam(list(siamese_net.parameters()) + list(classification_net_1.parameters()) + list(classification_net_2.parameters()), lr=0.001)
optimizer = torch.optim.Adam(list(siamese_net.parameters()), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)


# In[350]:


torch.set_printoptions(precision=3)
epochs_log = []
train_loss_log = []
val_loss_log = []
val_auc_log = []
val_auc_binary_log = []

#device = "cuda" if torch.cuda.is_available() else "cpu"

def train_siamese_network(siamese_net, train_loader, criterion_1, criterion_2, optimizer, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in train_loader:
            input1, input2, labels, binary_labels = data
            #input1, input2, labels, binary_labels = input1.to(device), input2.to(device), labels.to(device), binary_labels.to(device)
            output, binary_output = siamese_net(input1, input2)
            loss1 = criterion_1(output, labels)
            #print("Binary Output :", binary_output)
            #binary_output_probs = torch.sigmoid(binary_output)
            #print("Binary Sigmoid :", binary_output_probs)
            binary_labels_float = binary_labels.float()
            #print("Binary Labels :", binary_labels_float)
            #print("Binary Labels :", binary_labels_float.view(-1, 1))
            loss2 = criterion_2(binary_output, binary_labels_float.view(-1, 1))
            #loss2 = criterion_2(binary_output, binary_labels_float)
            #loss2 = criterion_2(binary_output, binary_labels)
            loss = 1.5 * loss1 + 0.5 * loss2 # Sum of all losses

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        roc_auc, val_loss, binary_roc_auc = roc_auc_val_loss(siamese_net, val_loader, criterion_1, criterion_2)
        # Ausgabe der aktuellen Lernrate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.3f}, Validation loss: {val_loss:.3f}, AUC: {roc_auc:.3f}, Binary_AUC: {binary_roc_auc:.3f}")
        # Ausgabe des Diagrams:
        epochs_log.append(epoch)
        train_loss_log.append(float(loss))
        val_loss_log.append(float(val_loss))
        val_auc_log.append(roc_auc)
        val_auc_binary_log.append(binary_roc_auc)


# Beispielaufruf der Funktion
num_epochs = 30
train_siamese_network(siamese_net, train_loader, criterion_1, criterion_2, optimizer, num_epochs)#.to(device)


# In[345]:


def roc_auc_val_loss(model, val_loader, criterion1, criterion2):
    model.eval()
    roc_auc_total = 0
    val_loss_total = 0
    roc_auc_total_binary = 0
    
    with torch.no_grad():
        for data in val_loader:
            input1, input2, labels, binary_labels = data
            output, binary_output = model(input1, input2)
            val_loss1 = criterion_1(output, labels)
            binary_labels_float = binary_labels.float()
            val_loss2 = criterion_2(binary_output, binary_labels_float.view(-1, 1))
            val_loss = 1.5 * val_loss1 + 0.5 * val_loss2
            val_loss_total += val_loss.item()
            output_probs = F.softmax(output, dim=1)  # Softmax Transformation
            roc_auc_total += roc_auc_score(labels.cpu().numpy(), output_probs.cpu().numpy(), average='weighted', multi_class='ovo')
            roc_auc_total_binary += roc_auc_score(binary_labels.cpu().numpy(), binary_output.cpu().numpy())
            
    return roc_auc_total / len(val_loader), val_loss_total / len(val_loader), roc_auc_total_binary / len(val_loader)


def get_predictions(model, test_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    all_predictions = []
    label_prediction = []

    with torch.no_grad():
        for data in test_loader:
            input1, input2, labels, binary_labels = data
            input1, input2, labels, binary_labels = input1.to(device), input2.to(device), labels.to(device), binary_labels.to(device)
            output, binary_output = model(input1, input2)
            output_probs = F.softmax(output, dim=1)  # Softmax 
            _, predicted = torch.max(output_probs, 1)
            print("Pred: ", predicted)
            print("Real: ", labels)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            output_probs_rounded = torch.round(output_probs * 1000) / 1000  # Runden auf drei Stellen hinter dem Komma
            all_predictions.append(output_probs_rounded)
            label_prediction.append(predicted)
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")
    return all_predictions, label_prediction


# In[351]:


from matplotlib.pylab import plt
from numpy import arange
 
# Plot and label the training and validation loss values
plt.plot(epochs_log, train_loss_log, label='Training Loss')
plt.plot(epochs_log, val_loss_log, label='Validation Loss')
plt.plot(epochs_log, val_auc_log, label='Validation AUC')
plt.plot(epochs_log, val_auc_binary_log, label='Binary AUC')
 
# Add in a title and axes labels
plt.title('Training Loss, Validation Loss, Validation AUC, Binary AUC')
plt.xlabel('Epochs')
plt.ylabel('Loss / AUC')
 
# Display the plot
plt.legend(loc='best')
plt.show()


# In[352]:


###TEST DATA
all_predictions, label_prediction = get_predictions(siamese_net, test_loader)


# In[ ]:


###EVAL DATA:


# In[353]:


eval_predictions, eval_labels = get_predictions(siamese_net, eval_loader)


# In[309]:


###SHOW IMAGES AND PREDICTIONS:
eval_graph_copy = transfer_predicted_labels_to_graph(eval_labels,eval_graph)
show_images_from_tensor(eval_graph_copy, img_dir_test)


# In[ ]:





# In[ ]:





# In[ ]:


###TO USE ROC_AUC Calculation for Batches, where not all classes are inside the batch

from sklearn.metrics import roc_auc_score

def roc_auc_val_loss(model, val_loader, criterion):
    model.eval()
    roc_auc_total = 0
    val_loss_total = 0
    num_batches = len(val_loader)

    with torch.no_grad():
        for data in val_loader:
            input1, input2, labels = data
            output = model(input1, input2)
            val_loss = criterion(output, labels)
            val_loss_total += val_loss.item()
            
            # Check if all classes are present in the batch
            if len(torch.unique(labels)) == output.shape[1]:
                output_probs = F.softmax(output, dim=1)  # Softmax Transformation
                roc_auc_total += roc_auc_score(labels.cpu().numpy(), output_probs.cpu().numpy(), average='weighted', multi_class='ovo')
            else:
                num_batches -= 1  # Reduce the count of valid batches
            
    # Calculate the average AUC only for batches with all classes present
    if num_batches > 0:
        avg_roc_auc = roc_auc_total / num_batches
    else:
        avg_roc_auc = float('nan')  # If no valid batches found, return NaN
    
    return avg_roc_auc, val_loss_total / len(val_loader)

