import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

warnings.filterwarnings('ignore')

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

model_name = "target_model_hate_explain"

batch_size = 64
num_epochs = 20
learning_rate = 2e-5
num_labels = 2

tokenizer = AutoTokenizer.from_pretrained('roberta-base')

def tokenize(example):
    encoded = tokenizer(example["post_text"], truncation=True, padding='max_length', max_length=512)
    encoded["labels"] = [int(label) for label in example['label']]
    return encoded

df_train = pd.read_csv("../data/hatexplain_train.csv")
df_dev   = pd.read_csv("../data/hatexplain_val.csv")
df_test  = pd.read_csv("../data/hatexplain_test.csv")

cols_to_drop = ["Race", "Religion", "Gender", "Sexual Orientation", "Miscellaneous"]
df_train = df_train.drop(columns=cols_to_drop)

train_dataset = Dataset.from_pandas(df_train, preserve_index=False).shuffle(seed=42)
dev_dataset   = Dataset.from_pandas(df_dev,   preserve_index=False)
test_dataset  = Dataset.from_pandas(df_test,  preserve_index=False)

full_ds = DatasetDict({'train': train_dataset, 'dev': dev_dataset, 'test': test_dataset})

print("Tokenizing the dataset...")
tokenized_ds = full_ds.map(tokenize, batched=True)

train_dataset = tokenized_ds['train']
validation_dataset = tokenized_ds['dev']
test_dataset = tokenized_ds['test']

model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base', 
    num_labels=2,
    ignore_mismatched_sizes=True
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {"accuracy": accuracy, "f1": f1}

class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels.long())
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir=model_name,
    eval_strategy="steps",
    eval_steps=10,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to='none',
    fp16=True,
    gradient_accumulation_steps=8
)

trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()

print("\nEvaluating on test set...")
metrics = trainer.evaluate(eval_dataset=test_dataset)
print("Test Metrics:", metrics)

pred_logits = trainer.predict(test_dataset).predictions
pred_labels = np.argmax(pred_logits, axis=1)
true_labels = np.array(test_dataset['labels'])

cm = confusion_matrix(true_labels, pred_labels)
print("\nConfusion Matrix:\n", cm)

test_df = pd.DataFrame(test_dataset)

test_df['predicted_label'] = pred_labels
test_df['true_label'] = true_labels

misclassified_samples = test_df[test_df['predicted_label'] != test_df['true_label']]

os.makedirs('misclassified_samples', exist_ok=True)
misclassified_samples.to_csv('misclassified_samples/target_model.csv', index=False)

print("\nMisclassified samples saved to 'misclassified_samples/target_model.csv'.")
