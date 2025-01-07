import torch
from transformers import (AutoTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding)
from transformers import EarlyStoppingCallback
from datasets import Dataset, load_dataset, DatasetDict, load_from_disk
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import random
import os

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

warnings.filterwarnings("ignore")

model_name = "./concept_model_he"
batch_size = 64
num_epochs = 15
learning_rate = 2e-5
num_labels = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch.nn as nn

class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, logits, labels):
        loss = self.bce(logits, labels)
        weighted_loss = loss * self.weights
        return weighted_loss.mean()
        
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

def tokenize(example):
    student_inputs = tokenizer(example['post_text'], truncation=True, max_length=512)
    labels = torch.tensor([
        example['Race'], 
        example['Religion'], 
        example['Gender'], 
    ], dtype=torch.float32)
    labels = labels.T 
    return {
        'input_ids': student_inputs['input_ids'],
        'attention_mask': student_inputs['attention_mask'],
        'labels': labels
    }

df_train = pd.read_csv("../data/hatexplain_train.csv")
df_dev = pd.read_csv("../data/hatexplain_val.csv")
df_test = pd.read_csv("../data/hatexplain_test.csv")

df_train = df_train.drop(columns=['label', 'post_id', 'Miscellaneous', 'Sexual Orientation'])
df_dev = df_dev.drop(columns=['label', 'post_id', 'Miscellaneous', 'Sexual Orientation'])
df_test = df_test.drop(columns=['label', 'post_id', 'Miscellaneous', 'Sexual Orientation'])

labels = np.array(df_train[['Race', 'Religion', 'Gender']])

class_weights = []
for i in range(num_labels):
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels[:, i]),
        y=labels[:, i]
    )
    print(weights)
    class_weights.append(weights[1])  
class_weights[2] = class_weights[2] * 2.0 

from torch.utils.data import WeightedRandomSampler

def create_weighted_sampler(dataset, label_idx=2):
    gender_labels = np.array(dataset['Gender'])
    class_sample_count = np.array([len(np.where(gender_labels == t)[0]) for t in [0, 1]])
    weight_per_class = 1.0 / class_sample_count
    sample_weights = np.array([weight_per_class[int(g)] for g in gender_labels])
    sample_weights = torch.from_numpy(sample_weights).float()
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

class_weights = torch.tensor(class_weights).to(device)

dataset = Dataset.from_pandas(df_train)
dev_dataset = Dataset.from_pandas(df_dev)
test_dataset = Dataset.from_pandas(df_test)

full_ds = DatasetDict({'train': dataset, 'dev': dev_dataset, 'test': test_dataset})
train_sampler = create_weighted_sampler(dataset)

tokenized_ds = full_ds.map(tokenize, batched=True)
print("Tokenized dataset saved to disk.")

tokenized_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

print(f"Tokenized dataset example: {tokenized_ds['train'][0]}")

train_dataset = tokenized_ds['train']
validation_dataset = tokenized_ds['dev']
test_dataset = tokenized_ds['test']

model = RobertaForSequenceClassification.from_pretrained(
    'target_model_hate_explain/checkpoint-150',
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
    predictions = (predictions > 0.5).astype(int)
    accuracy = (predictions == labels).mean()
    f1 = f1_score(labels, predictions, average='macro')
    return {"f1": f1, "accuracy": accuracy}

training_args = TrainingArguments(
    output_dir=model_name,
    eval_strategy="steps",
    eval_steps=50,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    logging_dir='./logs',
    save_total_limit=1,
    save_steps=100,  
    load_best_model_at_end=True,  
    metric_for_best_model="f1",
    logging_steps=100,
    report_to='none',
    fp16=True
)

def get_train_dataloader_custom(trainer):
    train_sampler = create_weighted_sampler(trainer.train_dataset, label_idx=2)
    return torch.utils.data.DataLoader(
        trainer.train_dataset,
        batch_size=trainer.args.train_batch_size,
        sampler=train_sampler,
        collate_fn=trainer.data_collator,
        drop_last=False,
        pin_memory=True
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

trainer.get_train_dataloader = lambda: get_train_dataloader_custom(trainer)

trainer.train()

metrics = trainer.evaluate(eval_dataset=test_dataset)
print(f'Metrics:\n {metrics}')

predictions = trainer.predict(test_dataset).predictions
predictions = torch.sigmoid(torch.tensor(predictions)).cpu().numpy()
predictions = (predictions > 0.5).astype(int)

labels = np.array(test_dataset['labels'])

print("Classification Report:\n")
print(classification_report(labels, predictions, target_names=[f'Label {i}' for i in range(num_labels)]))

def plot_confusion_matrix(y_true, y_pred, labels, label_idx, class_name, output_dir="./concept_model_cm"):
    print(y_true.shape, y_pred.shape)
    cm = confusion_matrix(y_true[:, label_idx], y_pred[:, label_idx])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/confusion_matrix_{class_name}.png")
    plt.show()
    plt.close()  
    
CONCEPT_LABELS = ['Race', 'Religion', 'Gender']
for i, class_name in enumerate(CONCEPT_LABELS):
    plot_confusion_matrix(labels, predictions, labels, i, class_name)
