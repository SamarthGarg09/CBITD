import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from concept_gradient_v2 import ConceptGradients
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CONCEPT_LABELS = ['obscene', 'threat', 'sexual_explicit', 'insult', 'identity_attack']

ALL_WORD_LISTS = {
    'Personal Insults, Hate Speech, and Misogyny': [
        "dumb",
        "lecherous",
        "pussy",
        "leaker",
        "dummy",
        "whining",
        "sexist",
        "Butt's",
        "troll",
        "nincompoop",
        "prickly",
        "foul",
        "misogyny",
        "vile",
        "degenerate",
        "self-centered",
        "pathetic",
        "fool",
        "inferior",
        "obnoxious",
        "deplorable",
        "assholes",
        "jerks",
        "d-bag",
        "nincompoop"
    ],
    'Violence, Extremist Speech, and Political Aggression': [
        "BOMBS",
        "rifle",
        "handgun",
        "extremists",
        "kill",
        "terrorist",
        "assassination",
        "gun",
        "gunmericans",
        "Nazis",
        "war",
        "ISIS",
        "beheaded",
        "Rambo",
        "killing",
        "shot",
        "weapon",
        "bullets",
        "terror",
        "anarchists",
        "extremists",
        "killing spree",
        "militia",
        "extremist",
        "war",
        "destruction"
    ],
    'Racism, Sexism, Religious Intolerance, and Homophobia': [
        "Racist",
        "homophobic",
        "Islamophobia",
        "white supremacy",
        "neo-Nazis",
        "supremacist",
        "Jews",
        "Muslims",
        "Black",
        "minorities",
        "religion",
        "theology",
        "Islamic",
        "Christian",
        "sexism",
        "gender",
        "transphobic",
        "religious bigotry",
        "racial discrimination",
        "homophobia",
        "anti-semitism"
    ],
    'Hate Speech, Personal Attacks, and Derogatory Descriptions': [
        "dumb",
        "stupid",
        "trash",
        "filth",
        "liars",
        "scum",
        "jerk",
        "scumbag",
        "idiot",
        "moron",
        "hypocrite",
        "dirtbag",
        "frauds",
        "piece of trash",
        "asshole",
        "idiots",
        "delusional",
        "jerk",
        "disgraceful",
        "bigots",
        "tyrants",
        "hate",
        "vile",
        "filth",
        "disgusting"
    ],
    'Gender-based Insults, Misogyny, and Sexual Degradation': [
        "misogyny",
        "sexist",
        "pussy",
        "sexual",
        "lecherous",
        "filthy",
        "chauvinistic",
        "rape",
        "sluts",
        "degrading",
        "objectification",
        "sex",
        "sexist",
        "gender",
        "harassment",
        "rape",
        "sexual abuse",
        "degradation",
        "misogynistic"
    ]
}




CONCEPT_LABELS = ['obscene', 'threat', 'sexual_explicit', 'insult', 'identity_attack']

class X2YModel(nn.Module):
    def __init__(self, model_name='./saved_target_model', num_classes=2):
        super(X2YModel, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        if inputs_embeds is not None:
            outputs = self.model.roberta(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            outputs = self.model.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return self.model.classifier(outputs.last_hidden_state)

class X2CModel(nn.Module):
    def __init__(self, model_name='./saved_concept_model', num_concepts=5):
        super(X2CModel, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_name, num_labels=num_concepts, ignore_mismatched_sizes=True
        ).to(device)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        if inputs_embeds is not None:
            outputs = self.model.roberta(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            outputs = self.model.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return self.model.classifier(outputs.last_hidden_state)

def load_and_preprocess_data(word_list):
    df = pd.read_csv('./dataset/test.csv')

    # Filter based on provided word list
    pattern = '|'.join(word_list)
    df_filtered = df[df['comment_text'].str.contains(pattern, case=False, na=False)].copy()
    # Binary transformation for the 'toxicity' column
    df_filtered['toxicity'] = (df_filtered['toxicity'] > 0.5).astype(int)

    # Process other columns except 'obscene' and 'insult'
    for column in df_filtered.columns:
        if column not in ['comment_text', 'toxicity']:
            df_filtered[column] = (df_filtered[column] > 0.0).astype(int)
        elif column == 'toxicity':
            df_filtered[column] = (df_filtered[column] > 0.5).astype(int)

    # Drop unnecessary columns (including 'obscene', 'insult')
    columns_to_drop = CONCEPT_LABELS + ['severe_toxicity']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df_filtered.columns]
    df_filtered.drop(columns=existing_columns_to_drop, inplace=True)
    print("Remaining columns:", df_filtered.columns.tolist())
    print("Filtered DataFrame shape:", df_filtered.shape)
    return df_filtered


def setup_models_and_cg():

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    x2y_model = X2YModel().to(device)

    x2c_model = X2CModel().to(device)

    def forward_func(embeddings, attention_mask):
        return x2y_model(inputs_embeds=embeddings, attention_mask=attention_mask)

    def concept_forward_func(embeddings, attention_mask):
        return x2c_model(inputs_embeds=embeddings, attention_mask=attention_mask)

    cg = ConceptGradients(
        forward_func,
        concept_forward_func=concept_forward_func,
        x2y_model=x2y_model,
        x2c_model=x2c_model,
    )

    return tokenizer, x2y_model, x2c_model, cg


def process_data_and_calculate_gradients(df_filtered, word_list, tokenizer, x2y_model, x2c_model, cg):
    results = []
    for idx, row in tqdm(df_filtered.iterrows(), total=df_filtered.shape[0]):
        sentence = row['comment_text']
        true_label = int(row['toxicity'])
        try:
            result = calculate_concept_gradient_for_sentence(
                sentence, tokenizer, x2y_model, x2c_model, cg, true_label
            )
            
            if result['predicted_class'] != true_label:
                result['true_class'] = true_label
                results.append(result)
                
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
    
    return results

def calculate_concept_gradient_for_sentence(sentence, tokenizer, x2y_model, x2c_model, cg, label):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        embeddings = x2y_model.model.get_input_embeddings()(input_ids)
        logits = x2y_model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=-1).cpu()
        target_index = torch.argmax(probs, dim=-1).item()

    embeddings.requires_grad_(True)
    attention_mask = attention_mask.float()
    attention_mask.requires_grad_(True)

    attr = cg.attribute(
        (embeddings, attention_mask),
        mode='chain_rule_independent',
        target=label,
        n_concepts=5,
        target_layer_name='roberta.encoder.layer.11.output.dense',
        concept_layer_name='roberta.encoder.layer.11.output.dense',
    )

    with torch.no_grad():
        concept_logits = x2c_model(input_ids=input_ids, attention_mask=attention_mask)
        concept_probs = torch.sigmoid(concept_logits).cpu()

    predicted_class = torch.argmax(probs, dim=-1).item()
    concept_gradient = attr[0].detach().cpu().numpy()

    return {
        "sentence": sentence,
        "concept_gradient_mean": concept_gradient,
        "concept_probs": concept_probs[0].numpy(),
        "target_probs": probs[0].numpy(),
        "predicted_class": predicted_class,
        'target_idx': target_index
    }

def create_final_dataframe(results):
    df = pd.DataFrame(results)
    concept_probs_df = pd.DataFrame(df['concept_probs'].tolist(), columns=[f"{label}_prob" for label in CONCEPT_LABELS])
    flattened_gradients = [arr.flatten() for arr in df['concept_gradient_mean']]
    concept_gradients_df = pd.DataFrame(flattened_gradients, columns=[f"{label}" for label in CONCEPT_LABELS])
    final_df = pd.concat([df.drop(['concept_probs', 'concept_gradient_mean'], axis=1), concept_probs_df, concept_gradients_df], axis=1)
    return final_df

def save_profanity_counts(df_train, word_list_name, word_list):
    profanity_counts = {}
    for word in word_list:
        word_pattern = rf'\b{word}\b'
        count = df_train.comment_text.str.contains(word_pattern, case=False, na=False, regex=True).sum()
        profanity_counts[word] = count
    df_profanity_counts = pd.DataFrame(list(profanity_counts.items()), columns=['Profanity_Word', 'Count'])
    
    # Save the profanity counts DataFrame
    output_folder = f'output/{word_list_name}/profanity_counts'
    os.makedirs(output_folder, exist_ok=True)
    df_profanity_counts.to_csv(f'{output_folder}/{word_list_name}_profanity_counts.csv', index=False)

    return df_profanity_counts

def create_wordcloud(df_profanity_counts, word_list_name):
    wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(dict(df_profanity_counts.values))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    output_folder = f'output/{word_list_name}/word_clouds'
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(f'{output_folder}/{word_list_name}_wordcloud.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_top_concepts(final_df, concept_labels, word_list_name):
    concept_grad_columns = [f'{label}' for label in concept_labels]
    avg_concept_grad_scores = final_df[concept_grad_columns].mean()
    top_concepts = avg_concept_grad_scores.nlargest(5)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_concepts.index, y=top_concepts.values)
    # plt.title(f"{word_list_name}")
    plt.xlabel("Concept", fontsize=20)  # Increase font size for X-axis label
    plt.ylabel("WCA scores", fontsize=20)  # Increase font size for Y-axis label
    plt.xticks(rotation=45, fontsize=18)  # Increase font size for tick labels
    plt.yticks(fontsize=18)  # Increase font size for Y-axis ticks
    plt.tight_layout()
    output_folder = f'output/{word_list_name}/concept_plots'
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(f'{output_folder}/Hist_{word_list_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()

df_train = pd.read_csv('dataset/train.csv')

for word_list_name, word_list in ALL_WORD_LISTS.items():
    df_filtered = load_and_preprocess_data(word_list)
    df_filtered.to_csv('final_output/' + word_list_name + '.csv', index=False)
    tokenizer, x2y_model, x2c_model, cg = setup_models_and_cg()
    print(f"Processing word list: {word_list_name}")
    
    # Step 1: Calculate concept gradients and final DataFrame
    results = process_data_and_calculate_gradients(df_filtered, word_list, tokenizer, x2y_model, x2c_model, cg)
    final_df = create_final_dataframe(results)

    # Step 2: Save profanity counts and generate word cloud
    df_profanity_counts = save_profanity_counts(df_train, word_list_name, word_list)
    create_wordcloud(df_profanity_counts, word_list_name)

    # Step 3: Plot top concepts
    plot_top_concepts(final_df, CONCEPT_LABELS, word_list_name)
import os
import shutil

# Define source root directory and destination directories
source_root = './output'
dest_wordclouds = './final_output/word_clouds'
dest_plots = './final_output/plots'
dest_csv = './final_output/csv_dumps'

# Create destination directories if they don't exist
os.makedirs(dest_wordclouds, exist_ok=True)
os.makedirs(dest_plots, exist_ok=True)
os.makedirs(dest_csv, exist_ok=True)

# Walk through each folder inside the source directory
for root, dirs, files in os.walk(source_root):
    for file in files:
        file_path = os.path.join(root, file)
        
        # Check and move word cloud files
        if 'wordcloud' in file.lower():
            shutil.move(file_path, os.path.join(dest_wordclouds, file))
        
        # Check and move plot files
        
        # Check and move CSV dump files
        elif file.endswith('.csv'):
            shutil.move(file_path, os.path.join(dest_csv, file))
        else:
            shutil.move(file_path, os.path.join(dest_plots, file))

print("All files have been transferred successfully!")
