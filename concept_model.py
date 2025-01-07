import torch
from transformers import (AutoTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding)
from transformers import EarlyStoppingCallback
from datasets import Dataset, load_dataset, DatasetDict, load_from_disk
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
import warnings
import random
import os
######################################################################
##################### SET SEED #######################
def set_seed(seed):
    # Python's built-in random generator
    random.seed(seed)
    
    # NumPy's random generator
    np.random.seed(seed)
    
    # PyTorch's random generator (CPU)
    torch.manual_seed(seed)
    
    # If you are using GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If you're using multiple GPUs
        
    # Make CUDA deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Example usage
set_seed(42)
######################################################################

warnings.filterwarnings("ignore")

# Configurations
model_name = "./saved_concept_model"
batch_size = 64
num_epochs = 5
learning_rate = 2e-5
num_labels = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch.nn as nn

class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, weights):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.weights = weights
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, logits, labels):
        loss = self.bce(logits, labels)
        weighted_loss = loss * self.weights
        return weighted_loss.mean()
        
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    
    student_inputs = tokenizer(example['comment_text'], truncation=True, max_length=512)
    
    labels = torch.tensor([
        example['obscene'], 
        example['sexual_explicit'], 
        example['identity_attack'], 
        example['insult'], 
        example['threat']
    ], dtype=torch.float32)
    labels = labels.T 
    
    return {
        'input_ids': student_inputs['input_ids'],
        'attention_mask': student_inputs['attention_mask'],
        'labels': labels
    }

df = pd.read_csv("dataset/train_big.csv")
df = df.drop(columns=['toxicity', 'severe_toxicity', 'id'])
df_dev = pd.read_csv("dataset/dev.csv")
df_dev = df_dev.drop(columns=['toxicity', 'severe_toxicity', 'id'])
df_test = pd.read_csv("dataset/test.csv")
df_test = df_test.drop(columns=['toxicity', 'severe_toxicity', 'id'])

labels = np.array(df[['obscene', 'sexual_explicit', 'identity_attack', 'insult', 'threat']])

class_weights = []
for i in range(num_labels):
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels[:, i]),
        y=labels[:, i]
    )
    class_weights.append(weights[1])  

class_weights = torch.tensor(class_weights).to(device)

dataset = Dataset.from_pandas(df)
dev_dataset = Dataset.from_pandas(df_dev)
test_dataset = Dataset.from_pandas(df_test)

full_ds = DatasetDict({'train': dataset, 'dev': dev_dataset, 'test': test_dataset})
# Check if the tokenized dataset already exists
if os.path.exists('tokenized_dataset_concept_model'):
    # Load the tokenized dataset from disk
    tokenized_ds = load_from_disk('tokenized_dataset_concept_model')
    print("Loaded tokenized dataset from disk.")
else:
    # Tokenize and save the dataset
    print("Tokenizing the dataset...")
    tokenized_ds = full_ds.map(tokenize, batched=True)
    tokenized_ds.save_to_disk('tokenized_dataset_concept_model')
    print("Tokenized dataset saved to disk.")
    
# tokenized_ds = full_ds.map(tokenize, batched=True)
tokenized_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

print(f"Tokenized dataset example: {tokenized_ds['train'][0]}")

train_dataset = tokenized_ds['train']
validation_dataset = tokenized_ds['dev']
test_dataset = tokenized_ds['test']

model = RobertaForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    ignore_mismatched_sizes=True,
    problem_type="multi_label_classification"
)

target_layer = 'roberta.encoder.layer.11'

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

for name, param in model.named_parameters():
    print(f"Layer: {name} | Requires_grad: {param.requires_grad}")
    
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import Trainer

class WeightedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.class_weights = kwargs.pop('class_weights')
        super().__init__(*args, **kwargs)
        self.loss_fn = WeightedBCEWithLogitsLoss(self.class_weights)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss
        
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.sigmoid(torch.tensor(predictions)).cpu().numpy()
    predictions = (predictions > 0.5).astype(int)  # Binarize the predictions
    accuracy = (predictions == labels).mean()
    f1 = f1_score(labels, predictions, average='macro')
    return {"f1": f1, "accuracy": accuracy}

training_args = TrainingArguments(
    output_dir="./results_big_train",
    eval_strategy="steps",
    eval_steps=4500,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    logging_dir='./logs',
    save_total_limit=1,
    save_steps=4500,  
    load_best_model_at_end=True,  
    metric_for_best_model="f1",
    logging_steps=4500,
    report_to='none',
    fp16=True
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    class_weights=class_weights 
)

# trainer.train()

metrics = trainer.evaluate(eval_dataset=test_dataset)
print(f'Metrics:\n {metrics}')

predictions = trainer.predict(test_dataset).predictions
predictions = torch.sigmoid(torch.tensor(predictions)).cpu().numpy()
predictions = (predictions > 0.5).astype(int)

labels = np.array(test_dataset['labels'])

print("Classification Report:\n")
print(classification_report(labels, predictions, target_names=[f'Label {i}' for i in range(num_labels)]))

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, labels, label_idx, class_name, output_dir="./confusion_matrices"):
    cm = confusion_matrix(y_true[:, label_idx], y_pred[:, label_idx])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/confusion_matrix_{class_name}.png")
    plt.show()
    plt.close()  
    
CONCEPT_LABELS = ['obscene', 'threat', 'sexual_explicit', 'insult', 'identity_attack']
for i, class_name in enumerate(CONCEPT_LABELS):
    plot_confusion_matrix(labels, predictions, labels, i, class_name)
model.save_pretrained("./saved_concept_model")
tokenizer.save_pretrained("./saved_concept_model")