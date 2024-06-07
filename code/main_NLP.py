#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 15:36:53 2023

@author: yair
"""

import torch
from data import read_imdb_split, IMDbDataset
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW


train_texts, train_labels = read_imdb_split(r'C:\Users\gabri\Desktop\practical_transformers\data\aclImdb\train')
test_texts, test_labels = read_imdb_split(r'C:\Users\gabri\Desktop\practical_transformers\data\aclImdb\test')


train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts,
                                                                    train_labels,
                                                                    test_size=.2)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)

train_dataset = torch.utils.data.Subset(train_dataset, range(100))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    print(f'Epoch {epoch+1}')
    for i_batch, batch in enumerate(train_loader):
        if i_batch%10==0:
            print(f'Batch {i_batch+1}/{len(train_loader)}')
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()

print('Training Ended')

model.eval()

test_dataset = torch.utils.data.Subset(test_dataset, range(1000))
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

total_loss = 0
for i_batch, batch in enumerate(test_loader):
    if i_batch%10==0:
        print(f'Batch {i_batch+1}/{len(train_loader)}')
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs[0]
    total_loss += loss
    print(loss, total_loss)

