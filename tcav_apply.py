import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from datasets import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from tcav import TCAV

# Define the X2YModel class
class X2YModel(nn.Module):
    def __init__(self, model_name='saved_target_model', num_classes=2):
        super(X2YModel, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
        
    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        if inputs_embeds is not None:
            outputs = self.model.roberta(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            input_ids = input_ids.long()
            with torch.no_grad():
                input_embeds = self.model.roberta.get_input_embeddings()(input_ids)
            outputs = self.model.roberta(inputs_embeds=input_embeds, attention_mask=attention_mask)
        return self.model.classifier(outputs.last_hidden_state)

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["comment_text"], padding="max_length", truncation=True)

def prepare_labels(examples, label_columns):
    labels = []
    for i in range(len(examples[label_columns[0]])):
        labels.append([float(examples[column][i]) for column in label_columns])
    return {"labels": labels}

def calculate_pos_weight(dl):
    pos_cnt = None
    cnt = 0
    with torch.no_grad():
        for batch in dl:
            if pos_cnt is None:
                pos_cnt = batch['labels'].sum(0).clone()
            else:
                pos_cnt += batch['labels'].sum(0).clone()
            cnt += batch['labels'].shape[0]
    neg_cnt = cnt - pos_cnt
    pos_weight = neg_cnt / pos_cnt
    return pos_weight

def main():
    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model
    x2y_model = X2YModel().to(device)
    
    # Load the dataset
    df = pd.read_csv('dataset/train.csv')
    dataset = Dataset.from_pandas(df)
    
    # Initialize the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
    # Prepare labels
    label_columns = ["obscene", "threat", "insult", "identity_attack", "sexual_explicit"]
    tokenized_dataset = tokenized_dataset.map(lambda x: prepare_labels(x, label_columns), batched=True)
    
    # Set the format for PyTorch
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Split the dataset
    train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    
    # Define layer names for TCAV
    layer_names = ['model.roberta.encoder.layer.11.output.dense']
    
    # Calculate positive weights
    hparams = {
        'task': 'classification',
        'n_epochs': 10,
        'patience': 10,
        'batch_size': 1024,
        'lr': 1e-3,
        'weight_decay': 1e-2,
        'pos_weight': None,
        'num_workers':4
    }
    train_dl = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=False)
    pos_weights = calculate_pos_weight(train_dl)
    hparams['pos_weight'] = pos_weights
    
    # Initialize TCAV
    tcav = TCAV(x2y_model, layer_names=layer_names, cache_dir='cav')
    
    # Generate random CAVs and CAVs
    tcav.generate_random_CAVs(train_dataset, test_dataset, n_repeat=1, force_rewrite_cache=True)
    tcav.generate_CAVs(train_dataset, test_dataset, n_repeats=1, hparams=hparams, force_rewrite_cache=True)

if __name__ == "__main__":
    main()
