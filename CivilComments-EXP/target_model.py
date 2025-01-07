import torch
from torch import nn
from transformers import (AutoTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding)
from datasets import Dataset, load_dataset, DatasetDict, load_from_disk
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F
import pandas as pd
import yaml
import warnings
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings
from imblearn.over_sampling import RandomOverSampler
import torch
import numpy as np
import random
import os

##################### SET SEED #######################
def set_seed(seed):
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# model_name = "roberta-base"
model_name = "saved_target_model_aug"
# model_name = 'saved_target_model'

batch_size = 64
num_epochs = 20
learning_rate = 2e-5
num_labels = 2

tokenizer = AutoTokenizer.from_pretrained('roberta-base')

# def tokenize(example):
#     student_inputs = tokenizer(example['comment_text'], truncation=True, max_length=512)
#     return {
#         'input_ids': student_inputs['input_ids'],
#         'attention_mask': student_inputs['attention_mask'],
#         'labels': example['toxicity']
#     }

def tokenize(example):
    student_inputs = tokenizer(example['comment_text'], padding='max_length', truncation=True, max_length=512)
    return {
        'input_ids': student_inputs['input_ids'],
        'attention_mask': student_inputs['attention_mask'],
        'labels': [int(label) for label in example['toxicity']] # Ensure labels are in integer format
    }



df = pd.read_csv("dataset/train_big.csv")
df_aug = pd.read_csv("dataset/augmented_toxic_no_profane_train.csv").tail(1000)

df_dev = pd.read_csv("dataset/dev.csv")
# df_test = pd.read_csv("dataset/test.csv")
df_test = pd.read_csv("dataset/toxicity_en.csv") #surge ai
df_test.columns = ['comment_text', 'toxicity']
df_test['toxicity'] = df_test['toxicity'].map({'Toxic':1, 'Non-Toxic':0})


df = df.dropna(subset=['toxicity'])
df_dev = df_dev.dropna(subset=['toxicity'])
df_test = pd.read_csv("dataset/toxicity_en.csv")
df_test.columns = ['comment_text', 'toxicity']

df_test = df_test.dropna(subset=['toxicity'])
df_test['toxicity'] = df_test['toxicity'].map({'Toxic': 1, 'Not Toxic': 0})

df_test = df_test.dropna(subset=['toxicity'])
# print(df.columns)
# print(df_test.columns)

label_0_samples = df[df['toxicity'] == 0].sample(2000, random_state=42)
label_1_samples = df[df['toxicity'] == 1].sample(1000, random_state=42)

balanced_df = pd.concat([label_0_samples, label_1_samples])

balanced_df = pd.concat([balanced_df, df_aug])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
balanced_dataset = Dataset.from_pandas(balanced_df)

dataset = balanced_dataset.shuffle(seed=42).select(range(4000))
dev_dataset = Dataset.from_pandas(df_dev)
test_dataset = Dataset.from_pandas(df_test)
test_dataset = test_dataset.shuffle().select(range(200))

full_ds = DatasetDict({'train': dataset, 'dev': dev_dataset, 'test': test_dataset})

# Check if the tokenized dataset already exists
# if os.path.exists('tokenized_dataset'):
#     # Load the tokenized dataset from disk
#     tokenized_ds = load_from_disk('tokenized_dataset')
#     print("Loaded tokenized dataset from disk.")
# else:
    # Tokenize and save the dataset
print("Tokenizing the dataset...")
tokenized_ds = full_ds.map(tokenize, batched=True)
# tokenized_ds.save_to_disk('tokenized_dataset')
# print("Tokenized dataset saved to disk.")

train_dataset = tokenized_ds['train']
validation_dataset = tokenized_ds['dev']
test_dataset = tokenized_ds['test']

model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    return {"accuracy": accuracy, "f1": f1}

class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss().to(self.model.device)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="steps",
    eval_steps=1000,
    # max_steps=100, 
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=1000,
    save_steps=1000,  
    load_best_model_at_end=True, 
    metric_for_best_model="f1",
    report_to='none',
    fp16=True,
    gradient_accumulation_steps=8
)
from transformers import EarlyStoppingCallback

trainer = WeightedLossTrainer(
# trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# trainer.train()

trainer.evaluate(eval_dataset=test_dataset)
metrics = trainer.evaluate(eval_dataset=test_dataset)
print('Metrics:\n', metrics)
predictions = trainer.predict(test_dataset).predictions
predictions = predictions.argmax(axis=1)
labels = test_dataset['labels']

cm = confusion_matrix(labels, predictions)
print("Confusion Matrix:\n", cm)
predictions = trainer.predict(test_dataset).predictions

predictions = predictions.argmax(axis=1)

test_df = pd.DataFrame(test_dataset)

test_df['predicted_label'] = predictions
test_df['true_label'] = test_df['labels']

misclassified_samples = test_df[test_df['predicted_label'] != test_df['true_label']]

print("Misclassified samples:\n", misclassified_samples)

misclassified_samples.to_csv('misclassified_samples.csv', index=False)

print("Misclassified samples saved to 'misclassified_samples.csv'.")

# model.save_pretrained("./saved_target_model_aug")
# tokenizer.save_pretrained("./saved_target_model_fake")