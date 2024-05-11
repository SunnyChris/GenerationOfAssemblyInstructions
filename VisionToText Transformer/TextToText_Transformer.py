#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datasets import DatasetDict, Dataset
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import BertForSequenceClassification, BertTokenizer


# In[2]:


train_eng_fra = 'data/bauteile_training/train.txt'
val_eng_fra = 'data/bauteile_training/val.txt'
num_sequences = 'data/bauteile_training/labels.txt'
#test_eng_fra = 'data/bauteile_training/test.txt'


# In[3]:


# Definition des Modells und des Tokenizers
model_checkpoint = "distilbert-base-uncased"
num_labels = 4  # Anzahl der Klassen (2 für 2 oder 3 Montageanweisungen)
#model_sequences = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
#tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# In[4]:


# Speichern des Modells
#model_sequences.save_pretrained("data/bauteile_training/model1")
# Speichern des Tokenizers
#tokenizer.save_pretrained("data/bauteile_training/tokenizer1")

# Laden des Modells
model_sequences = AutoModelForSequenceClassification.from_pretrained("data/bauteile_training/model1")
# Laden des Tokenizers
tokenizer = AutoTokenizer.from_pretrained("data/bauteile_training/tokenizer1")


# In[5]:


class LanguageDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        with open(dataset, "r",  encoding="utf8") as f:
            content = f.readlines()
        self.content = content
        
    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        english_sentence = []
        french_sentence = []
        line_dict = {}
        line = self.content[idx] #print(line)
        line = line.strip().split("	")
        english_sentence = line[0]
        english_sentence = english_sentence[:-1]
        french_sentence = line[1]
        if french_sentence[-1:] == '!':
            french_sentence =  french_sentence[:-2]
        else:
            french_sentence =  french_sentence[:-1]
        return {"input_ids":tokenizer(english_sentence)["input_ids"], "labels":tokenizer(french_sentence)["input_ids"]}


# In[6]:


class NumerationDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        with open(dataset, "r",  encoding="utf8") as f:
            content = f.readlines()
        self.content = content
        
    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        english_sentence = []
        french_sentence = []
        line_dict = {}
        line = self.content[idx] #print(line)
        line = line.strip().split(";")
        english_sentence = line[0]
        french_sentence = line[1]
        input_ids = tokenizer(english_sentence, truncation=True, padding="max_length", max_length=50, return_tensors="pt")["input_ids"]
        labels = int(french_sentence)
        return {"input_ids": input_ids.squeeze(), "labels": torch.tensor(labels)}


# In[7]:


# Erstellen des Trainingsdatasets
train_dataset_sequences = NumerationDataset(num_sequences)
print(train_dataset_sequences[1])


# In[8]:


# Trainingsfunktion
def train_model(train_data, model):
    training_args = TrainingArguments(
        output_dir="./results",  # Ausgabeverzeichnis für das Training
        per_device_train_batch_size=8, ###Vorher = 2
        num_train_epochs=20,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
    )
    trainer.train()


# In[9]:


# Training des Modells
train_model(train_dataset_sequences, model_sequences)


# In[10]:


torch.save(model_sequences, 'model_sequences.pth')
torch.save(model_sequences.state_dict(), 'model_sequences_weights.pth')


# In[11]:


# Inferenzfunktion
def predict_instruction_count(input_text):
    model = torch.load('model_sequences.pth')
    model.load_state_dict(torch.load('model_sequences_weights.pth'))
    model_checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model.to('cpu')
    model.eval()
    inputs = tokenizer(input_text, truncation=True, padding="max_length", max_length=50, return_tensors="pt")
    with torch.no_grad():  # Deaktivieren von Gradientenberechnungen
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()
    return predicted_label #+ 2  # Da die Labels 0 und 1 sind, fügen wir 2 hinzu, um auf die Anzahl der Montageanweisungen zu kommen (2 oder 3)


# In[12]:


# Beispielinferenz
input_text = "Gartenstuhlgestell, Sitzkissen, Rückenkissen"
predicted_count = predict_instruction_count(input_text)
print(f"Predicted Instruction Count: {predicted_count}")


# In[17]:


# Laden des Modells und des Tokenizers
model_checkpoint = "t5-small"
model_prediction = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# In[18]:


# Speichern des Modells
#model_prediction.save_pretrained("data/bauteile_training/model2")
# Speichern des Tokenizers
#tokenizer.save_pretrained("data/bauteile_training/tokenizer2")

# Laden des Modells
#model_prediction = AutoModelForSequenceClassification.from_pretrained("./data/bauteile_training/model2")
# Laden des Tokenizers
#tokenizer = AutoTokenizer.from_pretrained("data/bauteile_training/tokenizer2")


# In[19]:


batch_size = 16  ### default 128
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-xsum",
    evaluation_strategy = "epoch",
    learning_rate=2e-3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=20, ###default: 50
    predict_with_generate=True,
    push_to_hub=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_prediction)


# In[20]:


trainer = Seq2SeqTrainer(
    model_prediction,
    args,
    train_dataset=LanguageDataset(train_eng_fra),
    eval_dataset=LanguageDataset(val_eng_fra),
    data_collator=data_collator,
    tokenizer=tokenizer,
)

#model = model.to(device)

trainer.train()


# In[ ]:


torch.save(model_prediction, 'model_prediction.pth')
torch.save(model_prediction.state_dict(), 'model_prediction_weights.pth')


# In[ ]:


#preds = model.generate(input_ids = torch.tensor(test[index]["input_ids"]).to(device).view(1,-1))
#print(tokenizer.decode(np.array(preds.cpu()[0]))[6:-4])
# Funktion zur Generierung von Vorhersagen für eine Eingabesequenz mit Abtasten
def generate_predictions(input_text, num_return_sequences, top_k=5, temperature=1.5):
    model = torch.load('model_prediction.pth')
    model.load_state_dict(torch.load('model_prediction_weights.pth'))
    model_checkpoint = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model.to('cpu')
    model.eval()
    input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        outputs = model.generate(input_ids, 
                                  num_return_sequences=num_return_sequences, 
                                  max_length=50, 
                                  early_stopping=False, 
                                  do_sample=True, 
                                  top_k=top_k,  # Top-K-Sampling
                                  temperature=temperature)  # Temperatur beim Sampling
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]


# In[ ]:


def generate_unique_predictions(input_text, num_return_sequences, similarity_threshold=0.7):
    predictions_set = set()
    unique_predictions = []
    while len(unique_predictions) < num_return_sequences:
        predictions = generate_predictions(input_text, num_return_sequences - len(unique_predictions))
        for prediction in predictions:
            # Überprüfen, ob der Satz eindeutig ist und nicht zu ähnlich zu einem bereits vorhandenen Satz ist
            unique = True
            for existing_prediction in unique_predictions:
                # Berechnen der Ähnlichkeit der Wörter der Vorhersage mit den bereits vorhandenen Vorhersagen
                words_prediction = set(prediction.split())
                words_existing_prediction = set(existing_prediction.split())
                intersection = words_prediction.intersection(words_existing_prediction)
                similarity = len(intersection) / max(len(words_prediction), len(words_existing_prediction))
                if similarity >= similarity_threshold:
                    unique = False
                    break
            if unique:
                unique_predictions.append(prediction)
                if len(unique_predictions) == num_return_sequences:
                    break
    return unique_predictions

input_text = "Türblatt, Türgriff, Scharniere"
num_return_sequences = predict_instruction_count(input_text)
predictions = generate_unique_predictions(input_text, num_return_sequences)

print("Input: ", input_text)
print("Predictions:")
for i, pred in enumerate(predictions):
    print(f"  Prediction {i+1}: {pred}")


# In[ ]:




