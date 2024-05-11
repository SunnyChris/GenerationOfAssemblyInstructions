#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###DATA Augmentation


# In[ ]:


import os
from PIL import Image

def addnoise(im):
    noise = Image.effect_noise(im.size, 50)
    return noise

###for filename in os.listdir(directory):
    
directory = 'dataVisionToText/toydataset/rawtest'
for count, filename in enumerate(os.listdir(directory)):
    file = os.path.join(directory, filename)
    #print(file)
    image = Image.open(file).convert("RGB")
    ###img1 = image.resize((272, 204))
    img1 = image.resize((1156, 867))
    img2 = img1.convert("L")
    img3 = img1.convert("1")
    ###count = count + 93 + 81
    ###count = count + 93
    convert = str(count)
    img1.save('dataVisionToText/toydataset/evaluation/' + convert + "-image.jpg")
    #img2.save('dataVisionToText/toydataset/images/' + convert + "-2.jpg")
    #img3.save('dataVisionToText/toydataset/images/' + convert + "-3.jpg")
    ##addnoise(image)
    ###display(image)
    ###display(img2)
    ###display(img3)
    ##display(noise)


# In[ ]:


###https://bipinkrishnan.github.io/ml-recipe-book/image_captioning.html


# In[1]:


import pandas as pd
from pathlib import Path

# create an empty dataframe with 'imgs' column
df = pd.DataFrame(columns=['imgs'])
# we will store the image files and captions here before putting it into dataframe
imgs, captions = [], []
# directory where the dataset is present
root_dir = Path("dataVisionToText/toydataset")


# In[2]:


# get the contents of 'captions.txt' file
with open(root_dir/"captions.txt", "r") as f:
    content = f.readlines()


# In[3]:


for line in content:
    line = line.strip().split(",")

    # extract the required informations
    img_path = line[0]
    caption = line[1]
    
        ####print(caption)

    # store the image path
    imgs.append(root_dir/"images"/img_path)
        ###print(img_path)
    # store the caption
    captions.append(caption)
        ###print(caption)


# In[4]:


df.loc[:, 'imgs'] = imgs
df.loc[:, 'captions'] = captions


# In[5]:


from transformers import AutoFeatureExtractor, AutoTokenizer

encoder_checkpoint = "google/vit-base-patch16-224-in21k"
decoder_checkpoint = "gpt2"

feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)


# In[6]:


tokenizer.pad_token = tokenizer.eos_token


# In[7]:


from PIL import Image

# maximum length for the captions
max_length = 256 ###default 128
sample = df.iloc[130]

# sample image
image = Image.open(sample['imgs']).convert('RGB')
# sample caption
caption = sample['captions']

# apply feature extractor on the sample image
inputs = feature_extractor(images=image, return_tensors='pt')
# apply tokenizer
outputs = tokenizer(
            caption, 
            max_length=max_length, 
            truncation=True, 
            padding='max_length',
            return_tensors='pt',
        )

display(image)


# In[8]:


print(inputs)
print(outputs)


# In[9]:


from torch.utils.data import Dataset

class LoadDataset(Dataset):
    def __init__(self, df):
        self.images = df['imgs'].values
        self.captions = df['captions'].values
    
    def __getitem__(self, idx):
        # everything to return is stored inside this dict
        inputs = dict()

        # load the image and apply feature_extractor
        image_path = str(self.images[idx])
        image = Image.open(image_path).convert("RGB")
        image = feature_extractor(images=image, return_tensors='pt')

        # load the caption and apply tokenizer
        caption = self.captions[idx]
        labels = tokenizer(
            caption, 
            max_length=max_length, 
            truncation=True, 
            padding='max_length',
            return_tensors='pt',
        )['input_ids'][0]
        
        # store the inputs(pixel_values) and labels(input_ids) in the dict we created
        inputs['pixel_values'] = image['pixel_values'].squeeze()   
        inputs['labels'] = labels
        return inputs
    
    def __len__(self):
        return len(self.images)


# In[10]:


### Install: pip install -U scikit-learn scipy matplotlib

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)  ###default testsize=0.2


# In[11]:


train_ds = LoadDataset(train_df)
test_ds = LoadDataset(test_df)


# In[12]:


###Training


# In[13]:


from transformers import VisionEncoderDecoderModel

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    encoder_checkpoint, 
    decoder_checkpoint
)


# In[14]:


model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id


# In[15]:


# set number of beams for beam search to 4
num_beams = 4 ### default 4
model.config.num_beams = num_beams


# In[16]:


from transformers import Seq2SeqTrainingArguments

# batch size
bs = 8  ### Default 8

training_args = Seq2SeqTrainingArguments(
    output_dir="image-caption-generator", # name of the directory to store training outputs
    evaluation_strategy="epoch",          # evaluate after each epoch
    per_device_train_batch_size=bs,       # batch size during training
    per_device_eval_batch_size=bs,        # batch size during evaluation
    learning_rate=5e-5,
    weight_decay=0.01,                    # weight decay parameter for AdamW optimizer
    num_train_epochs=25,                   # number of epochs to train        #### Default 5
    save_strategy='epoch',                # save checkpoints after each epoch
    report_to='none',                     # prevent reporting to wandb, mlflow...
)


# In[17]:


from transformers import Seq2SeqTrainer, default_data_collator

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=feature_extractor,
    data_collator=default_data_collator,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    args=training_args,
)

trainer.train()


# In[18]:


###Testing - Inference


# In[19]:


inputs = test_ds[80]['pixel_values']


# In[20]:


sample = test_df.iloc[80]
image = Image.open(sample['imgs']).convert('RGB')
display(image)


# In[21]:


import torch

model.eval()
with torch.no_grad():
    # uncomment the below line if feature extractor is not applied to the image already
    # inputs = feature_extractor(images=inputs, return_tensors='pt').pixel_values

    # generate caption for the image
    out = model.generate(
        inputs.unsqueeze(0).to('cuda'), # move inputs to GPU
        num_beams=num_beams, max_new_tokens = 100 
        )

# convert token ids to string format
decoded_out = tokenizer.decode(out[0], skip_special_tokens=True)

print(decoded_out)


# In[22]:


###Evaluation - Inference


# In[38]:


###inputs = dict()

# store the inputs(pixel_values) in the dict we created
###inputs['pixel_values'] = image['pixel_values'].squeeze()

# load the image and apply feature_extractor
image = Image.open('dataVisionToText/toydataset/evaluationSMALL/7-image.jpg').convert("RGB")
display(image)
###image = feature_extractor(images=image, return_tensors='pt')


# In[39]:


import torch

model.eval()
with torch.no_grad():
    # uncomment the below line if feature extractor is not applied to the image already
    inputs = feature_extractor(images=image, return_tensors='pt').pixel_values

    # generate caption for the image
    out = model.generate(
        inputs.to('cuda'), # move inputs to GPU
        num_beams=num_beams, max_new_tokens = 100
        )

# convert token ids to string format
decoded_out = tokenizer.decode(out[0], skip_special_tokens=True)

print(decoded_out)


# In[ ]:




