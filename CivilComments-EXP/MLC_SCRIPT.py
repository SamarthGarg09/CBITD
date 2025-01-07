import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch import nn
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TOKENIZERS_PARALLELISM'] = "false"

df = pd.read_csv('dataset/train.csv')
dev = pd.read_csv('dataset/dev.csv')
test = pd.read_csv('dataset/test.csv')
df.head()

pd.set_option('display.max_colwidth', None)
df.sample(5)

df.drop(columns=['id', 'severe_toxicity', 'toxicity'], inplace=True)
dev.drop(columns=['id', 'severe_toxicity', 'toxicity'], inplace=True)
test.drop(columns=['id', 'severe_toxicity', 'toxicity'], inplace=True)
df.shape

cols = df.columns[1:]
cols

NUM_LABELS=5
MAX_LENGTH=512
BATCH_SIZE=64
MODEL_NAME='saved_target_model'
LR=2e-5
NUM_WARMUP_STEPS=3600
NUM_TRAIN_EPOCHS=5
NUM_LOG_STEPS=2250
NUM_SAVE_STEPS=2250

class ToxicityDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = df[['obscene', 'threat', 'sexual_explicit', 'insult', 'identity_attack']].values
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        comment = self.df.iloc[idx].comment_text
        inputs = self.tokenizer(
            comment,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

model_name = 'roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = ToxicityDataset(df, tokenizer, 128)
dataset[0]

train_dataset = ToxicityDataset(df, tokenizer, MAX_LENGTH)
dev_dataset = ToxicityDataset(dev, tokenizer, MAX_LENGTH)
test_dataset = ToxicityDataset(test, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

len(train_loader)

model = RobertaForSequenceClassification.from_pretrained("saved_target_model", num_labels=NUM_LABELS, problem_type="multi_label_classification", ignore_mismatched_sizes=True)
model.to('cuda')
target_layer = 'roberta.encoder.layer.10'

unfreeze = False
for name, layer in model.named_modules():
    if name == target_layer:
        unfreeze = True
    if not unfreeze:
        for param in layer.parameters():
            param.requires_grad = False
    else:
        for param in layer.parameters():
            param.requires_grad = True  
            
criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=LR)
NUM_TRAIN_STEPS=len(train_loader)*NUM_TRAIN_EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=NUM_WARMUP_STEPS, num_training_steps=NUM_TRAIN_STEPS)

TOTAL_STEPS = len(train_loader) * NUM_TRAIN_EPOCHS
TOTAL_STEPS

import torch
import numpy as np
from tqdm import tqdm

TOTAL_STEPS = len(train_loader) * NUM_TRAIN_EPOCHS
loss_i = []
val_loss, val_acc = [], []
saved_models = []
step = 0
train_iter = iter(train_loader)

for step in tqdm(range(TOTAL_STEPS)):
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        batch = next(train_iter)
    
    model.train()
    optimizer.zero_grad()
    batch = {k: v.to('cuda') for k, v in batch.items()}
    logits = model(batch['input_ids'], attention_mask=batch['attention_mask']).logits
    loss = criterion(logits, batch['labels'])
    loss.backward()
    optimizer.step()
    scheduler.step()
    loss_i.append(loss.item())
    
    if step % NUM_LOG_STEPS == 0 and step > 0:
        model.eval()
        total_val_loss = 0
        correct_predictions = np.zeros(len(cols))
        total_predictions = np.zeros(len(cols))
        
        with torch.no_grad():
            for batch in tqdm(dev_loader, leave=False, total=len(dev_loader)):
                batch = {k: v.to('cuda') for k, v in batch.items()}
                logits = model(batch['input_ids'], attention_mask=batch['attention_mask']).logits
                loss = criterion(logits, batch['labels'])
                total_val_loss += loss.item()
                preds = torch.sigmoid(logits).cpu().numpy()
                labels = batch['labels'].cpu().numpy()
                preds = (preds > 0.5)
                correct_predictions += np.sum(preds == labels, axis=0)
                total_predictions += labels.shape[0]
        
        val_loss_epoch = total_val_loss / len(dev_loader)
        accuracy = correct_predictions / total_predictions
        val_loss.append(val_loss_epoch)
        val_acc.append(accuracy)
        print(f"Step {step} | Training Loss: {loss_i[-1]:.4f} | Validation Loss: {val_loss_epoch:.4f} | Validation Accuracy: {accuracy}")
        model.train()
    
    if step % NUM_SAVE_STEPS == 0 and step > 0:
        if len(saved_models) < 2 or val_loss_epoch < max(saved_models, key=lambda x: x[1])[1]:
            model_save_path = f'model_checkpoint_step_{step}.pt'
            torch.save(model.state_dict(), model_save_path)
            saved_models.append((model_save_path, val_loss_epoch))
            saved_models = sorted(saved_models, key=lambda x: x[1])[:2]
            print(f"Model checkpoint saved at {model_save_path}")

from collections import defaultdict

model.eval()
test_loss = 0
correct_predictions = np.zeros(len(cols))  
total_predictions=np.zeros(len(cols))
samples = defaultdict(list)
with torch.no_grad():
    for batch in tqdm(test_loader):
        batch = {k:v.to('cuda') for k,v in batch.items()}
        logits = model(batch['input_ids'], attention_mask=batch['attention_mask']).logits
        loss = criterion(logits, batch['labels'])
        test_loss += loss.item()
        preds = torch.sigmoid(logits).cpu().numpy()
        labels = batch['labels'].cpu().numpy()
        preds = (preds > 0.5)
        correct_predictions += (preds==labels).sum(axis=0)
        total_predictions += labels.shape[0]
        decoded_text = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        samples['text'].extend(decoded_text)
        samples['labels'].extend(labels)
        samples['preds'].extend(preds)
        
    total_loss = test_loss / len(test_loader)
    total_accuracy = correct_predictions / total_predictions
    print(f"Test Loss: {total_loss:.4f} | Test Accuracy: {total_accuracy}")

import random

random_100 = random.sample(list(zip(samples['text'], samples['labels'], samples['preds'])), 100)
dump = pd.DataFrame(random_100, columns=['text', 'labels', 'preds'])
dump.to_csv('random_100.csv', index=False)
