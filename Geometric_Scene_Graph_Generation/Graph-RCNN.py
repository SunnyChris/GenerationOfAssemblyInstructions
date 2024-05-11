#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###extract_object_proposals:


# In[16]:


import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
import json

# Laden des Faster R-CNN-Modells vortrainiert auf COCO-Datensatz
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Laden des COCO-Datensatzes zur Konvertierung von Klassen-IDs in Klassennamen
with open('instances_val2017.json', 'r') as f:
    coco_data = json.load(f)

# Extrahieren des Klassen-IDs-zu-Namen-Mappings
class_id_to_name = {}
for category in coco_data['categories']:
    class_id_to_name[category['id']] = category['name']

# Definieren einer Funktion zum Extrahieren von Objektvorschlägen und Klassenverteilungen
def extract_object_proposals_and_class_distributions(image):
    # Transformieren des Bildes für das Modell
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Hinzufügen einer Batch-Dimension

    # Durchführen der Vorhersage mit dem Modell
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)

    # Extrahieren der Objektvorschläge aus den Vorhersagen
    proposals = predictions[0]['boxes']  # 'boxes' enthält die begrenzenden Kästen der Vorschläge
    class_scores = predictions[0]['scores']  # 'scores' enthält die Klassenverteilungen der Vorschläge
    classes = predictions[0]['labels']  # 'labels' enthält die Klassen-IDs der Vorschläge

    # Erstellen einer Matrix für die Klassenverteilungen
    class_distributions = torch.zeros(len(proposals), max(classes) + 1)  # Initialisieren mit Nullen

    # Aktualisieren der Klassenverteilungen für erkannte Klassen
    for i, class_id in enumerate(classes):
        class_distributions[i, class_id] = class_scores[i]

    return proposals, classes, class_scores, class_distributions


# Beispiel Verwendung
# Hier das Bild laden
image_path = 'fussball.jpg'
image = Image.open(image_path)

# Extrahieren der Objektvorschläge, Klassen und Klassenverteilungen
object_proposals, class_ids, class_scores, class_distributions = extract_object_proposals_and_class_distributions(image)

# Konvertieren der Klassen-IDs in Klassennamen
class_names = [class_id_to_name[class_id.item()] for class_id in class_ids]

# Anzeigen der extrahierten Objektvorschläge, Klassen und Klassenverteilungen
print("Class Scores Shape:", class_scores.shape)
print("Class Distributions Shape:", class_distributions.shape)
print("Object Proposals:", object_proposals)
print("Classes:", class_names)
print("Class Scores:", class_scores)
print("Class Distributions:", class_distributions)


# In[17]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def visualize_object_proposals(image_path, object_proposals):
    # Bild laden
    img = np.array(Image.open(image_path))

    # Bounding Boxes visualisieren
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for box in object_proposals:
        # Extrahieren der Koordinaten der Bounding Box
        x_min, y_min, x_max, y_max = box

        # Berechnen der Breite und Höhe der Bounding Box
        width = x_max - x_min
        height = y_max - y_min

        # Erstellen und Hinzufügen der Bounding Box zum Plot
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()

# Bounding Boxes visualisieren
visualize_object_proposals(image_path, object_proposals)


# In[ ]:


###RelationProposalNetwork


# In[18]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationProposalNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RelationProposalNetwork, self).__init__()
        self.proj_subject = nn.Linear(input_dim, hidden_dim)
        self.proj_object = nn.Linear(input_dim, hidden_dim)
        self.mlp = nn.Linear(hidden_dim * 2, 1)  # Multi-Layer Perceptron für die Bewertung der Beziehungen
        self.sigmoid = nn.Sigmoid()

    def forward(self, object_class_distributions):
        # Projektion der Objektklassenverteilungen für Subjekte und Objekte
        subject_embeddings = self.proj_subject(object_class_distributions)
        object_embeddings = self.proj_object(object_class_distributions)

        # Berechnung der Relatedness-Werte für alle Paare von Objekten
        n = object_class_distributions.size(0)
        relatedness_scores = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    concatenated_features = torch.cat([subject_embeddings[i], object_embeddings[j]], dim=0)
                    score = self.mlp(concatenated_features)
                    relatedness_scores[i][j] = score

        # Anwenden der Sigmoid-Funktion auf die Relatedness-Werte
        relatedness_scores = self.sigmoid(relatedness_scores)

        return relatedness_scores

# Beispiel Verwendung
input_dim = 54 ###TESTWERT: 10  # Dimension der Eingabemerkmale der Objekte (Anzahl der Klassen)
hidden_dim = 256  # Dimension der versteckten Merkmale für die Projektion

# Initialisieren des Relation Proposal Networks
repn = RelationProposalNetwork(input_dim, hidden_dim)

# Beispiel Eingabe: Objektklassenverteilungen (Po)
# Angenommen, wir haben 10 Objekte und jede Objektklassenverteilung hat eine Dimension von 10 (für 10 Klassen)
###object_class_distributions = torch.randn(10, input_dim)

# Durchführen des Forward-Passes durch das Relation Proposal Network
#relatedness_scores = repn(object_class_distributions)
#class_distributions = class_distributions.view(1, -1)
relatedness_scores = repn(class_distributions)

print("Relatedness Scores Shape:", relatedness_scores.shape)
print("Relatedness Scores:", relatedness_scores)


# In[ ]:


###Feature extraction for object_proposals, relatedness_scores


# In[22]:


def extract_features_from_relatedness_scores(object_proposals, relatedness_scores):
    object_features = []
    relationship_features = []

    num_objects = len(object_proposals)

    # Extrahieren der Objektmerkmale aus den Vorschlägen
    for proposal in object_proposals:
        # Annahme: Objektvorschläge sind begrenzende Kästen (z. B. xmin, ymin, xmax, ymax)
        object_feature = torch.tensor([proposal[0], proposal[1], proposal[2], proposal[3]])  # Beispiel: Nur die begrenzenden Kasten als Merkmale
        object_features.append(object_feature)

    # Extrahieren der Beziehungsmerkmale aus den Relatedness Scores
    for i in range(num_objects):
        for j in range(num_objects):
            if i != j:
                # Sicherstellen, dass wir nur einzigartige Paare betrachten
                if i < j:
                    # Konkatenation der Merkmale der Objektpaare
                    concatenated_features = torch.cat([object_features[i], object_features[j]], dim=0)
                    relationship_features.append(concatenated_features)

    # Konvertieren der Listen von Features in Tensoren
    object_features = torch.stack(object_features)
    relationship_features = torch.stack(relationship_features)

    return object_features, relationship_features

# Beispiel Verwendung
object_features, relationship_features = extract_features_from_relatedness_scores(object_proposals, relatedness_scores)

print("Object Features Shape:", object_features.shape)
print("Relationship Features Shape:", relationship_features.shape)
print("Object Features:", object_features)
print("Relationship Features:", relationship_features)


# In[ ]:


###AttentionalGCN mit Message Passing


# In[27]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionalGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AttentionalGCN, self).__init__()
        self.W_skip = nn.Linear(input_dim, output_dim)  # W_skip für Skip-Verbindungen
        self.W_sr = nn.Linear(input_dim, output_dim)    # W_sr für Relationship -> Object
        self.W_or = nn.Linear(input_dim, output_dim)    # W_or für Object -> Relationship
        self.W_rs = nn.Linear(input_dim, output_dim)    # W_rs für Object -> Relationship
        self.W_ro = nn.Linear(input_dim, output_dim)    # W_ro für Object -> Relationship
        
        self.Wa = nn.Linear(input_dim * 2, 1)  # 2-Schicht MLP für die Aufmerksamkeit
        self.softmax = nn.Softmax(dim=1)

    def forward(self, object_features, relationship_features, adjacency_matrix):
        # Konvertieren der Gewichte in den richtigen Datentyp
        self.W_skip.to(object_features.dtype)
        self.W_sr.to(relationship_features.dtype)
        self.W_or.to(object_features.dtype)
        self.W_rs.to(object_features.dtype)
        self.W_ro.to(relationship_features.dtype)
        self.Wa.to(object_features.dtype)
        
        # Message Passing von Objekten zu Objekten (nur zwischen benachbarten Objekten)
        messages_objects_to_objects = torch.matmul(adjacency_matrix, self.W_sr(relationship_features))
        
        # Message Passing von Beziehungen zu Beziehungen (nur zwischen benachbarten Beziehungen)
        messages_relationships_to_relationships = torch.matmul(adjacency_matrix.t(), self.W_rs(object_features))
        
        # Message Passing von Objekten zu Beziehungen (nur zwischen benachbarten Objekten)
        messages_objects_to_relationships = torch.matmul(adjacency_matrix, self.W_or(object_features))
        
        # Message Passing von Beziehungen zu Objekten (nur zwischen benachbarten Beziehungen)
        messages_relationships_to_objects = torch.matmul(adjacency_matrix.t(), self.W_ro(relationship_features))
        
        # Aggregation der Nachrichten
        aggregated_messages = messages_objects_to_objects + messages_relationships_to_relationships + messages_objects_to_relationships + messages_relationships_to_objects
        
        #print("Aggregated Messages Shape:", aggregated_messages.shape)
        
        # Berechnung der Aufmerksamkeit
        concatenated_features = torch.cat([object_features.unsqueeze(1).repeat(1, adjacency_matrix.size(1), 1), 
                                           relationship_features.unsqueeze(0).repeat(adjacency_matrix.size(0), 1, 1)], dim=2)
        attention_scores = self.Wa(concatenated_features).squeeze()
        attention_scores = attention_scores * adjacency_matrix
        attention = self.softmax(attention_scores)
        
        #print("Attention Shape:", attention.shape)
        
        # Anwenden der Aufmerksamkeit auf die aggregierten Nachrichten
        attention_aggregated_messages = attention.unsqueeze(-1) * aggregated_messages.unsqueeze(1)
        
        #print("Attention Aggregated Messages Shape:", attention_aggregated_messages.shape)
        test = updated_object_features = self.W_skip(object_features)
        test2 = attention_aggregated_messages.sum(dim=1)
        #print("Skip Object Features:", test.shape)
        #print("Aggregated Messages:", test2.shape)
        
        # Update der Objekt- und Beziehungsrepräsentationen
        updated_object_features = self.W_skip(object_features) + attention_aggregated_messages.sum(dim=1)
        updated_relationship_features = self.W_skip(relationship_features) + attention_aggregated_messages.sum(dim=1)
        
        #print("Updated Object Features Shape:", updated_object_features.shape)
        #print("Updated Relationship Features Shape:", updated_relationship_features.shape)
        
        return updated_object_features, updated_relationship_features

# Beispiel Verwendung
input_dim = 64
hidden_dim = 128
output_dim = 64
num_nodes = 10
adjacency_matrix = torch.randint(2, (num_nodes, num_nodes)).float()  # Wir initialisieren die Adjazenzmatrix als FloatTensor
object_features = torch.randn(num_nodes, input_dim)
relationship_features = torch.randn(num_nodes, input_dim)

# Initialisieren und Durchführen des Forward-Passes durch das Modell
agcn = AttentionalGCN(input_dim, hidden_dim, output_dim)
output_objects, output_relationships = agcn(object_features, relationship_features, adjacency_matrix)

print("Matrix Shape is:", adjacency_matrix.shape)
print("Matrix is:", adjacency_matrix)
print("Obj_Features are:", object_features)
print("Rel_Features are:", relationship_features)
print("Updated Obj_Features:", output_objects)
print("Updated Rel_Features:", output_relationships)


# In[ ]:





# In[ ]:


###Classifyer with trainer: Attention in Training


# In[29]:


import torch
import torch.nn as nn
import torch.optim as optim

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        output = self.softmax(x)
        return output

def train_model(object_features, relationship_features, adjacency_matrix, object_labels, predicate_labels, num_epochs=30, learning_rate=0.001):
    # Initialisierung des Modells und der Loss-Funktion
    agcn = AttentionalGCN(input_dim, hidden_dim, output_dim)
    object_classifier = Classifier(input_dim, num_object_classes)  # Beispielklassifizierungsmodell für Objekte
    predicate_classifier = Classifier(input_dim, num_predicate_classes)  # Beispielklassifizierungsmodell für Prädikate
    criterion = nn.CrossEntropyLoss()  # Multi-class Cross Entropy Loss

    # Initialisierung des Optimierers
    optimizer = optim.Adam(list(agcn.parameters()) + list(object_classifier.parameters()) + list(predicate_classifier.parameters()), lr=learning_rate)

    for epoch in range(num_epochs):
        # Forward-Pass durch das Modell
        updated_object_features, updated_relationship_features = agcn(object_features, relationship_features, adjacency_matrix)
        
        # Klassifizierung der Objekte und Prädikate
        object_predictions = object_classifier(updated_object_features)
        predicate_predictions = predicate_classifier(updated_relationship_features)
        
        # Berechnung der Loss
        object_loss = criterion(object_predictions, object_labels)
        predicate_loss = criterion(predicate_predictions, predicate_labels)
        total_loss = object_loss + predicate_loss
        
        # Zurückpropagierung und Optimierung
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Ausgabe des Loss für jede Epoche
        print(f'Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss.item()}')

    print('Training Finished!')

# Beispiel Verwendung der Trainingsfunktion
input_dim = 64
hidden_dim = 128
output_dim = 64
num_nodes = 10
num_object_classes = 10  # Anzahl der Objektklassen
num_predicate_classes = 5  # Anzahl der Prädikatklassen

# Beispiel Daten (zufällig generiert)
object_features = torch.randn(num_nodes, input_dim)
relationship_features = torch.randn(num_nodes, input_dim)
adjacency_matrix = torch.randint(2, (num_nodes, num_nodes)).float()  # Beispiel Adjazenzmatrix (zufällig generiert)
object_labels = torch.randint(0, num_object_classes, (num_nodes,))
predicate_labels = torch.randint(0, num_predicate_classes, (num_nodes,))

# Training des Modells
train_model(object_features, relationship_features, adjacency_matrix, object_labels, predicate_labels)