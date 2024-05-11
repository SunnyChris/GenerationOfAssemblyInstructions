#!/usr/bin/env python
# coding: utf-8

# In[1]:


###Libraries


# In[332]:


import torch
import random
import networkx as nx
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from torch.utils.data.dataset import random_split
from PIL import Image
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models
from torch.nn import Transformer
import torch.optim as optim
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
#from trax.models.reformer import ReformerEncoderLayer, ReformerEncoder
#from trax.models.reformer import ReformerDecoderLayer, ReformerDecoder
from transformers import LongformerModel, LongformerConfig
from torchvision.models import resnet50
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[333]:


###Arguments


# In[364]:


img_dir_train = "GNNdataset3/Train"
img_dir_eval = "GNNdataset3/Eval"
model = models.resnet50(pretrained=True)

max_size_flattend_matrices = 169
max_size_flattend_features = 3328

###13x13 Matrix
graph1 = torch.tensor([
  [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
  [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
  [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
  [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
  [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
])

###8x8 Matrix
graph2 = torch.tensor([
  [0, 1, 1, 1, 0, 0, 1, 1],
  [1, 0, 1, 1, 0, 0, 0, 0],
  [1, 0, 0, 0, 0, 0, 0, 0],
  [1, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 1, 0],
  [0, 0, 0, 0, 0, 0, 0, 1],
  [1, 0, 0, 0, 1, 0, 0, 0],
  [1, 0, 0, 0, 0, 1, 0, 0]
])

###9x9 Matrix
graph3 = torch.tensor([
  [0, 1, 1, 1, 1, 0, 0, 1, 0],
  [1, 0, 1, 1, 1, 0, 0, 0, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0],
  [1, 0, 0, 0, 0, 0, 0, 0, 0],
  [1, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 1, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 1],
  [1, 0, 0, 0, 0, 1, 0, 0, 0],
  [0, 1, 0, 0, 0, 0, 1, 0, 0]
])

###13x13 Matrix
labels1 = torch.tensor([
  [0, 1, 1, 1, 1, 0, 0, 0, 0, 3, 3, 0, 0],
  [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 3, 3],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
  [3, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
  [3, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
  [0, 3, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
  [0, 3, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0]
])

###8x8 Matrix
labels2 = torch.tensor([
  [0, 1, 1, 1, 0, 0, 3, 3],
  [1, 0, 1, 1, 0, 0, 0, 0],
  [1, 0, 0, 0, 0, 0, 0, 0],
  [1, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 2, 0],
  [0, 0, 0, 0, 0, 0, 0, 2],
  [3, 0, 0, 0, 2, 0, 0, 0],
  [3, 0, 0, 0, 0, 2, 0, 0]
])

###9x9 Matrix
labels3 = torch.tensor([
  [0, 1, 1, 1, 1, 0, 0, 3, 0],
  [1, 0, 1, 1, 1, 0, 0, 0, 3],
  [1, 0, 0, 0, 0, 0, 0, 0, 0],
  [1, 0, 0, 0, 0, 0, 0, 0, 0],
  [1, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 2, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 2],
  [3, 0, 0, 0, 0, 2, 0, 0, 0],
  [0, 3, 0, 0, 0, 0, 2, 0, 0]
])

###Graph und Label Eval_data
graph4 = graph1
labels4 = labels1


# In[86]:


###Data Loading


# In[87]:


import os
import random
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, output_dim=256):
        super(ResNetFeatureExtractor, self).__init__()
        self.resnet = resnet50(pretrained=True)
        # Entferne den letzten vollständig verbundenen Layer und den Klassifikations-Layer
        self.resnet.fc = nn.Identity()
        # Füge einen neuen linearen Layer hinzu, um die Ausgabe auf die gewünschte Dimension zu reduzieren
        self.linear = nn.Linear(2048, output_dim)

    def forward(self, image):
        with torch.no_grad():
            features = self.resnet(image.unsqueeze(0))
        reduced_features = self.linear(features.squeeze())
        return reduced_features

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_folders = sorted(os.listdir(root_dir))
        self.resnet_extractor = ResNetFeatureExtractor()

    def __len__(self):
        return len(self.class_folders)

    def __getitem__(self, idx):
        class_folder = self.class_folders[idx]
        class_path = os.path.join(self.root_dir, class_folder)
        image_files = sorted(os.listdir(class_path))
        random_image = random.choice(image_files)
        image_path = os.path.join(class_path, random_image)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Display image
        #plt.imshow(image.permute(1, 2, 0))
        #plt.axis('off')
        #plt.show()
        
        # Convert to vector using ResNet50
        image_vector = self.resnet_extractor(image)
        
        return image_vector

    def get_image_tensor(self):
        image_tensor = torch.stack([self[idx] for idx in range(len(self))])
        return image_tensor.detach().numpy()

# Beispielaufruf
root_dir = "GNNdataset3/Train/Graph3"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = ImageDataset(root_dir, transform)
image_array = dataset.get_image_tensor()
print("Image Tensor Shape:", image_array.shape)
size_first_dimension = image_array.shape[0]
print("Größe der ersten Dimension:", size_first_dimension)


# In[374]:


def random_mapping_from_adjacency_matrix(num_nodes, adjacency_matrix, label_matrix):
    # Erstellen des Pfadgraphen G mit 4 Knoten
    adjacency_matrix = adjacency_matrix.numpy()
    label_matrix = label_matrix.numpy()
    G = nx.path_graph(num_nodes)  # 0-1-2-3
    # Erstellen des Graphen G aus der gegebenen Adjazenzmatrix
    G = nx.from_numpy_array(adjacency_matrix)
    F = nx.path_graph(num_nodes)  # 0-1-2-3
    # Erstellen des Graphen G aus der gegebenen Adjazenzmatrix
    F = nx.from_numpy_array(label_matrix)

    # Zufälliges Mapping der Knoten erstellen
    nodes = list(G.nodes)
    random.shuffle(nodes)
    mapping = {nodes[i]: nodes[(i+1)%len(nodes)] for i in range(len(nodes))}

    # Die Knoten des Graphen gemäß des zufälligen Mappings umbenennen
    H = nx.relabel_nodes(G, mapping)
    E = nx.relabel_nodes(F, mapping)
    
    # Erstellen der nodelist für die Ausgabe
    nodelist = list(range(num_nodes))

    return mapping, nx.to_numpy_array(H, nodelist=nodelist), nx.to_numpy_array(E, nodelist=nodelist)

num_nodes = 8
adjacency_matrix = graph2
label_matrix = labels1
random_mapping, new_matrix, new_matrix2 = random_mapping_from_adjacency_matrix(num_nodes, adjacency_matrix, label_matrix)

print("Zufälliges Mapping:")
print(random_mapping)

print("\nNeue Adjazenzmatrix:")
print(new_matrix)

print("\nNeue Labelmatrix:")
print(new_matrix2)


# In[89]:


def resort_vectors(mapping, features):
    ###Order regarding first index:
    first_index = list(mapping.keys())
    sorted_vectors = []
    for index in mapping:
        sorted_vectors.append(features[index])
    sorted_vectors = np.array(sorted_vectors)
    
    ###Order regarding second index:
    second_index = [value for value in mapping.values()]
    sorted_vectors_list = [sorted_vectors[second_index.index(i)] for i in range(len(second_index))]
    
    #print("Feature-Vektoren:")
    #print(features)
    
    #print("Mapping:")
    #print(mapping)

    #print("Neu sortierte Feature-Vektoren:")
    #print(np.array(sorted_vectors_list))
    
    return np.array(sorted_vectors_list)

#feature_vectors = np.random.rand(8, 1)

#features = resort_vectors(random_mapping, image_array)
#print(features)


# In[370]:


import networkx as nx
import numpy as np
import random

def reduce_nodes_and_mapping_and_features(adjacency_matrix, num_nodes, features, label_matrix):
    # Erstellen des Pfadgraphen G mit 4 Knoten
    G = nx.path_graph(4)  # 0-1-2-3

    # Reduzieren der Anzahl der Knoten in der Adjazenzmatrix
    reduced_adjacency_matrix = adjacency_matrix[:num_nodes, :num_nodes]
    reduced_labels = label_matrix[:num_nodes, :num_nodes]

    # Reduzieren der Anzahl der Knoten im Graphen
    H = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)

    # Mapping zwischen alten und neuen Knotenlabels
    mapping = {old_label: new_label for old_label, new_label in zip(range(num_nodes), H.nodes)}
    
    ###Reduce Features:
    features = torch.tensor(features)
    reduced_features = features[:num_nodes]

    return mapping, reduced_adjacency_matrix, reduced_features, reduced_labels

# Beispielaufruf der Funktion
num_nodes = 3
adjacency_matrix = new_matrix
label_matrix = new_matrix2
mapping, reduced_matrix, reduced_features, reduced_labels = reduce_nodes_and_mapping_and_features(adjacency_matrix, num_nodes, features, label_matrix)

print("Mapping:")
print(mapping)

print("\nReduzierte Adjazenzmatrix:")
print(reduced_matrix)

print("\nReduzierte Labelmatrix:")
print(reduced_labels)

print("Features:")
print(reduced_features)


# In[91]:


###Verbinden der Graphen


# In[376]:


import numpy as np
import torch

# Funktion zum Auffüllen von Arrays mit Nullen
def fill_with_zeros(array, target_length):
    return np.pad(array, (0, target_length - len(array)), mode='constant')

# Initialisiere leere Listen für die Ausgabe
all_reduced_matrices = []
all_reduced_features = []
all_reduced_labels = []

for _ in range(3):
    # Wähle zufällig einen der Ordner aus
    root_dir = np.random.choice(["GNNdataset3/Train/Graph1", "GNNdataset3/Train/Graph2", "GNNdataset3/Train/Graph3"])

    # Lade Array von Bildern und erhalte Anzahl der Knoten
    dataset = ImageDataset(root_dir, transform)
    image_array = dataset.get_image_tensor()
    num_nodes = image_array.shape[0]

    # Wähle die Adjazenzmatrix basierend auf dem ausgewählten Ordner
    if root_dir == "GNNdataset3/Train/Graph1":
        adjacency_matrix = graph1
        label_matrix = labels1
    elif root_dir == "GNNdataset3/Train/Graph2":
        adjacency_matrix = graph2
        label_matrix = labels2
    elif root_dir == "GNNdataset3/Train/Graph3":
        adjacency_matrix = graph3
        label_matrix = labels3

    # Sortiere Adjazenzmatrix neu und erstelle Mapping
    random_mapping, new_matrix, new_matrix2 = random_mapping_from_adjacency_matrix(num_nodes, adjacency_matrix, label_matrix)

    # Sortiere Merkmalsvektoren neu
    features = resort_vectors(random_mapping, image_array)

    # Reduziere Matrix und Merkmale
    reduced_nodes = np.random.randint(num_nodes // 2, num_nodes)  # Zufällige Anzahl von Knoten zur Reduzierung wählen
    mapping, reduced_matrix, reduced_features, reduced_labels = reduce_nodes_and_mapping_and_features(new_matrix, reduced_nodes, features, new_matrix2)

    # Flattening der reduzierten Matrizen und Merkmale
    flattened_matrix = reduced_matrix.flatten()
    flattened_features = reduced_features.flatten()
    flattened_labels = reduced_labels.flatten()

    # Auffüllen der Flattened Arrays mit Nullen, um die Längen auszugleichen
    filled_matrix = fill_with_zeros(flattened_matrix, max_size_flattend_matrices)
    filled_features = fill_with_zeros(flattened_features, max_size_flattend_features)
    filled_labels = fill_with_zeros(flattened_labels, max_size_flattend_matrices)

    # Konvertierung von Numpy-Arrays in PyTorch-Tensoren
    tensor_filled_matrix = torch.tensor(filled_matrix)
    tensor_filled_features = torch.tensor(filled_features)
    tensor_filled_labels = torch.tensor(filled_labels)

    # Füge die Tensoren der entsprechenden Liste hinzu
    all_reduced_matrices.append(tensor_filled_matrix)
    all_reduced_features.append(tensor_filled_features)
    all_reduced_labels.append(tensor_filled_labels)

# Stacke alle reduzierten Matrizen und reduzierten Merkmale zu einem großen Tensor bzw. Vektor
stacked_matrices = torch.stack(all_reduced_matrices, dim=0)
stacked_features = torch.stack(all_reduced_features, dim=0)
stacked_labels = torch.stack(all_reduced_labels, dim=0)

# Ausgabe der Shapes der kombinierten Tensoren
print("Shape der kombinierten reduzierten Matrizen:", stacked_matrices.shape)
print("Shape der kombinierten reduzierten Merkmale:", stacked_features.shape)
print("Shape der kombinierten reduzierten Labels:", stacked_labels.shape)


# In[93]:


torch.save(stacked_matrices, 'GNNdataset3/stacked_matrices.pt')
#stacked_matrices = torch.load('GNNdataset3/stacked_matrices.pt')
torch.save(stacked_features, 'GNNdataset3/stacked_features.pt')
#stacked_features = torch.load('GNNdataset3/stacked_features.pt')
torch.save(stacked_labels, 'GNNdataset3/stacked_labels.pt')
#stacked_labels = torch.load('GNNdataset3/stacked_labels.pt')


# In[16]:


#Load Data


# In[334]:


# Konvertierung der Tensoren in ein Dataset
dataset = TensorDataset(stacked_features, stacked_matrices, stacked_labels)

# Festlegen der Größen für die Aufteilung
train_size = 0.8
val_size = 0.1
test_size = 0.1

# Bestimmen der Längen der Trainings-, Validierungs- und Testsets
train_length = int(train_size * len(dataset))
val_length = int(val_size * len(dataset))
test_length = len(dataset) - train_length - val_length

# Aufteilen des Datasets
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_length, val_length, test_length])

# Festlegen der Batch-Größe
batch_size = 32

# Erstellen der DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# In[ ]:


# Model


# In[22]:


import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2, num_heads=2, hidden_dim=16, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder
        encoder_layers = TransformerEncoderLayer(input_dim, num_heads, dim_feedforward=hidden_dim*4, dropout=dropout)
        self.encoder = TransformerEncoder(encoder_layers, num_layers)
        
        # Decoder
        decoder_layers = TransformerDecoderLayer(input_dim, num_heads, dim_feedforward=hidden_dim*4, dropout=dropout)
        self.decoder = TransformerDecoder(decoder_layers, num_layers)

        # Linear layer for output
        self.linear_out = nn.Linear(input_dim, output_dim)

    def forward(self, src, tgt):
        # Encoder
        src = src.view(src.size(0), -1)  # Flattening des Eingabevektors der Features
        src = src.unsqueeze(1).repeat(1, src.size(1), 1)  # Repeat for each token
        #print(src.shape)
        encoder_output = self.encoder(src)
        encoder_output = encoder_output.to(dtype=torch.float32)

        # Decoder
        print(tgt.shape)
        tgt = tgt.unsqueeze(-1).repeat(1, 1, self.input_dim)
        tgt = tgt.repeat(1, src.size(1), 1)  # Repeat for each token
        tgt = tgt.to(dtype=torch.float32)
        #print(tgt.shape)
        decoder_output = self.decoder(tgt, encoder_output)

        # Linear layer for output
        output = self.linear_out(decoder_output)

        return output


# In[335]:


import torch
import torch.nn as nn
from torch.nn import Transformer

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2, num_heads=4, hidden_dim=256):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.transformer = nn.Transformer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, tgt):
        # Encoder
        encoder_output = self.encoder(src)
        #print(encoder_output.shape)

        # Decoder
        decoder_output = self.decoder(tgt)
        #print(decoder_output.shape)
        
        # Transformer
        output = self.transformer(encoder_output, decoder_output)
        #print(output.shape)

        # Fully Connected Layer
        output = self.fc(output)
        sigmoid = nn.Sigmoid()
        probabilities = sigmoid(output)

        #print(output.shape)
        
        return probabilities


# In[336]:


#Initialisierung


# In[355]:


input_dim = 3328  # Anzahl der Features (Flattened)
output_dim = 169  # Dimension der Adjazenzmatrix (Flattened)
num_layers = 4
num_heads = 8 #2
hidden_dim = 1024 #16
#dropout = 0.1

model = TransformerModel(input_dim, output_dim, num_layers, num_heads, hidden_dim)
#criterion = nn.MSELoss()  # Beispielverlustfunktion, je nach Anforderungen anpassen
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)  #weight_decay=0.001
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=30, factor=0.9, verbose=True)


# In[356]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_model(model, train_loader, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for features, labels in train_loader:
            features = features.float()
            labels = labels.float()

            # Forward Pass
            outputs = model(features, labels)
            # Berechnung des Verlusts
            loss = criterion(outputs, labels)
            # Backward Pass, Optimierung, Scheduler und Nullen der Gradienten
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            running_loss += loss.item()
        
        # Ausgabe des Verlusts nach jedem Epoch
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}')

train_model(model, train_loader, epochs=100)


# In[357]:


def inference_on_loader(model, test_loader, device = 'cpu'):
    model.eval()
    model.to(device)
    predictions = []
    with torch.no_grad():
        for src, tgt in test_loader:
            # Erzeugung von Dummy-Labels
            tgt = torch.zeros_like(tgt)
            #print(tgt.shape)
            src = src.float()
            tgt = tgt.float()
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt)
            predictions.append(output.cpu())
    return predictions


# In[358]:


def compute_accuracy(predictions, test_loader):
    correct = 0
    total = 0
    for pred, (_, labels) in zip(predictions, test_loader):
        pred_labels = pred.round().to(torch.int)  # Annahme: die Vorhersage ist als Wahrscheinlichkeiten oder scores gegeben
        labels = labels.view(-1).to(torch.int)  # Annahme: die Labels sind bereits in flacher Form
        
        # Vergleiche die flachen Adjazenzmatrizen
        correct += (pred_labels.view(-1) == labels).sum().item()  # Aufsummieren der korrekten Vorhersagen
        total += labels.size(0)  # Aktualisieren der Gesamtanzahl
        
    accuracy = correct / total
    return accuracy


# In[359]:


predictions = inference_on_loader(model, test_loader, device)
accuracy = compute_accuracy(predictions, test_loader)
print(f'Accuracy: {accuracy:.4f}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


###Attentional Message Passing


# In[ ]:


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
            nn.Linear(1024, 256),     # Reduziere die Dimension auf 64
            nn.ReLU())
        
    def forward(self, x):
        x = self.fc(x)
        return x


# In[362]:


class AttentionalGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AttentionalGCN, self).__init__()
        self.W_objects = nn.Linear(input_dim, output_dim)
        self.W_Nobjects = nn.Linear(input_dim, output_dim)
        self.W_relationships = nn.Linear(input_dim, output_dim)
        self.W_skip = nn.Linear(input_dim, output_dim)
        self.Wa = nn.Linear(input_dim * 2, 1)
        self.softmax = nn.Softmax(dim=1)

        # Weights initialization
        self.W_objects.to(torch.float32)
        self.W_Nobjects.to(torch.float32)
        self.W_relationships.to(torch.float32)
        self.W_skip.to(torch.float32)
        self.Wa.to(torch.float32)

    def forward(self, object_features, relationship_features, adjacency_matrix):
        # Message Passing between objects
        messages_objects_to_objects = self.W_objects(object_features)
        
        # Message Passing between neighbouring objects
        messages_neighbouring_objects = torch.matmul(adjacency_matrix.t(), self.W_Nobjects(object_features))

        # Message Passing between relationships (from relationship to object)
        messages_relationships = self.W_relationships(relationship_features)

        # Aggregating messages
        aggregated_messages = messages_objects_to_objects + messages_neighbouring_objects + messages_relationships

        # Attention calculation
        concatenated_features = torch.cat([object_features.unsqueeze(1).repeat(1, adjacency_matrix.size(1), 1),
                                           relationship_features.unsqueeze(0).repeat(adjacency_matrix.size(0), 1, 1)], dim=2)
        attention_scores = self.Wa(concatenated_features).squeeze()
        attention_scores = attention_scores * adjacency_matrix
        attention = self.softmax(attention_scores)

        # Applying attention to aggregated messages
        attention_aggregated_messages = attention.unsqueeze(-1) * aggregated_messages.unsqueeze(1)

        # Updating object features
        updated_object_features = self.W_skip(object_features) + attention_aggregated_messages.sum(dim=1)

        return updated_object_features

# Beispiel Verwendung
input_dim = 3
hidden_dim = 4
output_dim = 3
num_nodes = 3
adjacency_matrix = torch.tensor([[0, 1, 0],
                                 [0, 0, 1],
                                 [1, 0, 0]], dtype=torch.float32)  # Als FloatTensor definiert
object_features = torch.tensor([[0.1, 0.2, 0.3],
                                [0.4, 0.5, 0.6],
                                [0.7, 0.8, 0.9]])
relationship_features = torch.tensor([[0.01, 0.02, 0.03],
                                      [0.04, 0.05, 0.06],
                                      [0.07, 0.08, 0.09]])

# Initialisieren und Durchführen des Forward-Passes durch das Modell
agcn = AttentionalGCN(input_dim, hidden_dim, output_dim)
output_objects = agcn(object_features, relationship_features, adjacency_matrix)

print("Updated Object Features:")
print(output_objects)


# In[361]:


class SiameseNetworkWithAttention(nn.Module):
    def __init__(self, num_classes, input_dim, hidden_dim, output_dim):
        super(SiameseNetworkWithAttention, self).__init__()
        # Separate CNN for feature extraction
        self.cnn = CNN()
        self.attention_gcn = AttentionalGCN(input_dim, hidden_dim, output_dim)
        self.fc1 = nn.Linear(256 * 2 + output_dim, 64)  # Combine input sizes of both images and GCN output
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, input1, input2, object_features, relationship_features, adjacency_matrix):
        output1 = self.cnn(input1)
        output2 = self.cnn(input2)
        # Concatenate the feature vectors
        combined_output = torch.cat((output1, output2), 1)
        
        # Perform attention message passing only between objects
        updated_object_features = self.attention_gcn(object_features, relationship_features, adjacency_matrix)
        
        # Concatenate GCN output with combined_output
        combined_output_gcn = torch.cat((combined_output, updated_object_features), 1)
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(combined_output_gcn))
        x = self.fc2(x)
        
        return x


# In[ ]:


###INITIATE MODEL:
# Instantiate Siamese network
siamese_net = SiameseNetworkWithAttention(num_classes = 4, 
                                          input_dim = 256, 
                                          hidden_dim = 1024, 
                                          output_dim = 256)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(siamese_net.parameters()), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)


# In[ ]:


def roc_auc_val_loss(model, val_loader, criterion1, criterion2):
    model.eval()
    roc_auc_total = 0
    val_loss_total = 0
    roc_auc_total_binary = 0
    
    with torch.no_grad():
        for data in val_loader:
            input1, input2, labels, binary_labels = data
            output = model(input1, input2)
            val_loss = criterion(output, labels)
            val_loss_total += val_loss.item()
            output_probs = F.softmax(output, dim=1)  # Softmax Transformation
            roc_auc_total += roc_auc_score(labels.cpu().numpy(), output_probs.cpu().numpy(), average='weighted', multi_class='ovo')
            
    return roc_auc_total / len(val_loader), val_loss_total / len(val_loader)


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


# In[ ]:


torch.set_printoptions(precision=3)
epochs_log = []
train_loss_log = []
val_loss_log = []
val_auc_log = []

#device = "cuda" if torch.cuda.is_available() else "cpu"

def train_siamese_network(siamese_net, train_loader, criterion_1, criterion_2, optimizer, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in train_loader:
            input1, input2, labels, binary_labels = data
            output, binary_output = siamese_net(input1, input2, object_features)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        roc_auc, val_loss = roc_auc_val_loss(siamese_net, val_loader, criterion)
        # Ausgabe der aktuellen Lernrate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.3f}, Validation loss: {val_loss:.3f}, AUC: {roc_auc:.3f}, Binary_AUC: {binary_roc_auc:.3f}")
        # Ausgabe des Diagrams:
        epochs_log.append(epoch)
        train_loss_log.append(float(loss))
        val_loss_log.append(float(val_loss))
        val_auc_log.append(roc_auc)


# Beispielaufruf der Funktion
num_epochs = 30
train_siamese_network(siamese_net, train_loader, criterion, optimizer, num_epochs)#.to(device)

