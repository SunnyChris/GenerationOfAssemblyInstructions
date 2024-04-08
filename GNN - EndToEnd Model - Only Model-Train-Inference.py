# Definiere das GNN für Link Prediction
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv, GraphConv, GATConv, GATv2Conv, GraphSAGE, SAGEConv ######
import torch.nn.functional as F #####
from torch.optim.lr_scheduler import ReduceLROnPlateau

epochs_log = []
train_loss_log = []
val_loss_log = []
val_auc_log = []

class CNNModel(nn.Module):
    def __init__(self, model, weights):
        super().__init__()
        ###Backbone Resnet50 - RETRAINED
        #model_retrained = torch.load(model)
        #model_retrained.load_state_dict(torch.load(weights))
        #self.resnet = model_retrained
        ###Backbone Resnet50 - PRETRAINED
        self.resnet = models.resnet50(pretrained=True)
        #self.resnet = models.resnet18(pretrained=True)
        # Einfrieren der Parameter bis zum Layer layer2.0.downsample.1.bias
        #freeze = True
        #for name, param in resnet50.named_parameters():
        #for name, param in model_retrained.named_parameters():
        #    if 'layer4.0.downsample.1.bias' in name: ### 'layer2.0.downsample.1.bias' /// fc.bias
        #        freeze = False
        #    if freeze:
        #        param.requires_grad = False
        ###Ohne Merkmalsreduzierung:
        #self.resnet.fc = nn.Identity() ###DEFAULT: Verwendung der gesamten Resnet Architektur ohne den letzten Ausgabelayer
        ###Mit Merkmalsreduzierung:
        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, 512),   # Reduziere die Dimension auf 512
            nn.ReLU(),
            nn.Linear(512, 64),     # Reduziere die Dimension auf 64
            nn.ReLU()
        )
        ###Mit Merkmalsreduzierung, mit weniger Layern:
        #self.features = nn.Sequential(*list(resnet.children())[:-2])  # Entferne die letzten 2 Schichten
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive Durchschnittspooling-Schicht
        #self.fc = nn.Linear(2048, 64)  # Merkmalsreduktionsschicht

    def forward(self, x):
        #x = self.flatten(x)
        #x = x.unsqueeze(1)  # Füge eine Kanaldimension hinzu
        #x = x.expand(-1, 3, -1, -1)  # Erweitere die Kanaldimension auf 3
        #x = x.unsqueeze(0)
        ### Ohne reduzierte Layer:
        x = self.resnet(x)
        ### Mit reduzierten Layern:
        #x = self.features(x)
        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        #x = self.fc(x)
        ### ENDE mit teduzierten Layern
        return x


# In[45]:


class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, out_channels2, out_channels3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.conv3 = GCNConv(out_channels, out_channels2)
        self.conv4 = GCNConv(out_channels2, out_channels3)
        ###For the decoder for 3 output classes:
        #self.fc = nn.Linear(decoder_out_channels = 1, num_classes = 3)

    def encode(self, x, edge_index):
        x = self.flatten(x)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training) #### ADDED
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index)
        return x

    def decode(self, x, edge_label_index):
        x = (x[edge_label_index[0]] * x[edge_label_index[1]]).sum(dim=-1)
        ### For 3 Output Classes:
        # Linear transformation to get logits for each class
        #x = self.fc(x)
        # Apply softmax activation to get probabilities for each class
        #x = F.softmax(x, dim=-1)
        return x


# In[46]:


# Definiere die End-to-End-Architektur
class EndToEndModel(nn.Module):
    def __init__(self, model, weights, in_channels, hidden_channels, out_channels, out_channels2, out_channels3):
        super().__init__()
        self.cnn = CNNModel(model, weights)
        #self.cnn2 = CNNModel(model2, weights2)
        self.gnn = GNNModel(in_channels, hidden_channels, out_channels, out_channels2, out_channels3)

    def forward(self, img, edge_index, edge_label_index):
        cnn_output = self.cnn(img)
        #cnn_output2 = self.cnn(shapes)
        #cat_output = torch.cat((words, cnn_output, cnn_output2), dim=1)
        #cat_output = torch.cat((words, cnn_output), dim=1)
        encoded_output = self.gnn.encode(cnn_output, edge_index)
        decoded_output = self.gnn.decode(encoded_output, edge_label_index)
        #print(F.softmax(gnn_output, dim=1))
        return decoded_output


# In[51]:


def train_link_predictor(
    model, loader, val_data, optimizer, criterion, n_epochs=100 ### before 150 / 350 and 50 on words
):

    for epoch in range(1, n_epochs + 1):
        for data in loader:
            model.train()
            optimizer.zero_grad()
            ###CALL MODEL:
            cnn_output = model.cnn(train_data.x)
            encoded_output = model.gnn.encode(cnn_output, train_data.edge_index)
            # sampling training negatives for every training epoch
            neg_edge_index = negative_sampling(
                    edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
                    num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

            edge_label_index = torch.cat([train_data.edge_label_index, neg_edge_index],dim=-1,)
            edge_label = torch.cat([train_data.edge_label,train_data.edge_label.new_zeros(neg_edge_index.size(1))], dim=0)
            ####
            decoded_output = model.gnn.decode(encoded_output, edge_label_index).view(-1)
            #print(F.softmax(gnn_output, dim=1))
            ###END OF MODEL CALL
            loss = criterion(decoded_output, edge_label)
            loss.backward()
            optimizer.step()
            print("Batch finished")

        val_auc, val_loss = eval_link_predictor(model, val_data, criterion)  ###### ADDED
        
        scheduler.step(val_loss) #### ADDED
        # Ausgabe der aktuellen Lernrate
        current_lr = optimizer.param_groups[0]['lr']
        
        #print(out.view(-1).sigmoid())
        
        print(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val Loss: {val_loss:.3f}, Val AUC: {val_auc:.3f}, LR: {current_lr}") #### ADDED
        epochs_log.append(epoch)
        train_loss_log.append(float(loss))
        val_loss_log.append(float(val_loss))
        val_auc_log.append(val_auc)

    return model


@torch.no_grad()
def eval_link_predictor(model, data, criterion):                                        ######### ADDED

    model.eval()
    cnn_output = model.cnn(data.x)
    encoded_output = model.gnn.encode(cnn_output, data.edge_index)
    decoded_output = model.gnn.decode(encoded_output, data.edge_label_index).view(-1).sigmoid()
    val_loss = criterion(decoded_output, data.edge_label)                                        ######### ADDED
    
    return roc_auc_score(data.edge_label.cpu().numpy(), decoded_output.cpu().numpy()), val_loss  ######### ADDED

def get_predictions(model, x, edge_label_index):
    device = "cpu"
    model.to(device)
    model.eval()
    prediction_edge_label_index = ([],
                                   [])
    prediction_edge_label_index = torch.Tensor(prediction_edge_label_index)
    prediction_edge_label_index = prediction_edge_label_index.int()
    with torch.no_grad():
        cnn_output = model.cnn(x)
        encoded_output = model.gnn.encode(cnn_output, prediction_edge_label_index)
        decoded_output = model.gnn.decode(encoded_output, edge_label_index).view(-1).sigmoid()
    print(decoded_output.shape)
    print(decoded_output)
    return decoded_output

def get_prediction_accuracy(decoded_output, edge_label):
    pred_list = []
    counter1 = 1
    counter2 = 0
    correct = 0
    
    for item in decoded_output:
        if item >= 0.5:
            pred_list.append(counter1)
        else:
            pred_list.append(counter2)
    next
    label_values = edge_label
    label_values = label_values.type(torch.int64)
    
    for count, i in enumerate(label_values):
        if pred_list[count] == int(label_values[count]):
            correct = correct + 1
    acc = int(correct) / int(len(edge_label))
    return acc


# In[48]:


#model = GCNLinkPrediction(2048, 256, 128, 64, 1)
#model = GCNLinkPrediction(64, 64, 32, 32, 2) ### (64, 64, 32, 32, 1)
model = EndToEndModel('customResnet.pth','customResnet_weights.pth', 64, 64, 32, 32, 32)
#model = EndToEndModel('customResnet.pth','customResnet_weights.pth', 256, 32, 16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.005, weight_decay=0.01) #LR default = 0.001 weight_decay=0.01
criterion = torch.nn.BCEWithLogitsLoss()
###For 3 classes output:
#criterion = torch.nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
model = train_link_predictor(model, loader, val_data, optimizer, criterion).to(device)