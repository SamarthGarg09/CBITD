from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import torch.nn as nn
from concept_gradient_v2 import ConceptGradients

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Define the X2YModel (toxicity classifier)
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

# Define the X2CModel (concept classifier)
class X2CModel(nn.Module):
    def __init__(self, model_name='./saved_concept_model', num_concepts=5):
        super(X2CModel, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_name, num_labels=num_concepts, ignore_mismatched_sizes=True
        ).to(device)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        if inputs_embeds is not None:
            outputs = self.model.roberta(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask
            )
        else:
            outputs = self.model.roberta(
                input_ids=input_ids, attention_mask=attention_mask
            )
        return self.model.classifier(outputs.last_hidden_state)

# Instantiate the models
x2y_model = X2YModel().to(device)
x2c_model = X2CModel().to(device)

# Define the forward functions
def forward_func(embeddings, attention_mask):
    output = x2y_model(inputs_embeds=embeddings, attention_mask=attention_mask)
    return output

def concept_forward_func(embeddings, attention_mask):
    output = x2c_model(inputs_embeds=embeddings, attention_mask=attention_mask)
    return output

# Instantiate the ConceptGradients object
cg = ConceptGradients(
    forward_func,
    concept_forward_func=concept_forward_func,
    x2y_model=x2y_model,
    x2c_model=x2c_model,
)

def calculate_concept_gradient_for_sentence(
    sentence, target_index=None, concept_index=None, mode='chain_rule_independent'
):
    # Tokenize the input sentence
    inputs = tokenizer(
        sentence, return_tensors='pt', truncation=True, max_length=512, padding='max_length'
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Get embeddings
    with torch.no_grad():
        embeddings = x2y_model.model.get_input_embeddings()(input_ids)
        
    embeddings.requires_grad_(True)
    attention_mask = attention_mask.float()
    attention_mask.requires_grad_(True)
    
    # Predict the target class if not provided
    if target_index is None:
        with torch.no_grad():
            logits = x2y_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1)
            target_index = torch.argmax(probs, dim=-1).item()
    
    # Calculate concept gradient
    attr = cg.attribute(
        (embeddings, attention_mask),
        mode=mode,
        target=target_index,
        target_concept=concept_index,
        n_concepts=5,
        target_layer_name='roberta.encoder.layer.11.output.dense',
        concept_layer_name='roberta.encoder.layer.11.output.dense',
    )
    
    # Get concept logits and probabilities
    with torch.no_grad():
        concept_logits = x2c_model(input_ids=input_ids, attention_mask=attention_mask)
        concept_probs = torch.sigmoid(concept_logits).cpu()
    
    # Get target logits and probabilities
    with torch.no_grad():
        target_logits = x2y_model(input_ids=input_ids, attention_mask=attention_mask)
        target_probs = torch.softmax(target_logits, dim=-1).cpu()
    
    # Get the predicted class
    predicted_class = torch.argmax(target_probs, dim=-1).item()
    
    # Convert probabilities to NumPy arrays if needed
    concept_probs = concept_probs.numpy()
    target_probs = target_probs.numpy()
    
    concept_labels = ['obscene', 'threat', 'sexual_explicit', 'insult', 'identity_attack']
    
    return {
        "sentence": sentence,
        "concept_gradient": attr[0].detach().cpu().numpy(),
        "concept_probs": concept_probs,
        "target_probs": target_probs,
        "predicted_class": predicted_class,
        "concept_labels": concept_labels,
    }

def main():
    print("Enter a sentence (or type 'exit' to quit):")
    while True:
        sentence = input("> ")
        target_idx = int(input("> "))
        if sentence.lower() == 'exit':
            print("Exiting the program.")
            break
        result = calculate_concept_gradient_for_sentence(sentence, target_index=target_idx)
        
        # Extract results
        concept_gradient = result['concept_gradient']
        concept_probs = result['concept_probs']
        target_probs = result['target_probs']
        predicted_class = result['predicted_class']
        concept_labels = result['concept_labels']
        
        # Print the results
        print(f"\nSentence: {sentence}")
        print(f"Predicted Class: {predicted_class} (0: Non-toxic, 1: Toxic)")
        print("Target Probabilities:")
        print(f"  Non-toxic: {target_probs[0][0]:.4f}")
        print(f"  Toxic: {target_probs[0][1]:.4f}")
        
        print("\nConcept Probabilities:")
        for label, prob in zip(concept_labels, concept_probs[0]):
            print(f"  {label}: {prob:.4f}")
        
        print("\nConcept Gradient Scores (Mean across tokens):")
        for label, grad in zip(concept_labels, concept_gradient.mean(axis=0)):
            print(f"  {label}: {grad:.4f}")
        print("\nEnter another sentence (or type 'exit' to quit):")

if __name__ == "__main__":
    main()