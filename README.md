# Concept-Based Interpretability for Text Detection

This project implements concept-based interpretability techniques (CG and TCAV) for analyzing text toxicity detection models.

## Results

### Concept Gradient (CG) Results
- Concept 1 Accuracy: 0.8827
- Concept 2 Accuracy: 0.8307 
- Concept 3 Accuracy: 0.7140
- Concept 4 Accuracy: 0.9433
- Concept 5 Accuracy: 0.7487

F1-scores:
- Concept 1: 0.8317
- Concept 2: 0.3172
- Concept 3: 0.3983
- Concept 4: 0.9430
- Concept 5: 0.3968

### TCAV Results
- Concept 1 Accuracy: 0.832
- Concept 2 Accuracy: 0.772
- Concept 3 Accuracy: 0.938
- Concept 4 Accuracy: 0.818
- Concept 5 Accuracy: 0.732

### Dataset Distribution
Label distribution across splits:
```
                 train    dev    test
obscene         117677  12971  32395
sexual_explicit  55183   6029  14910
threat          85340   9471  23370
insult          362424  40337 100487
identity_attack 173962  19375  48181
toxicity        85052   9442  23381
```

### Target Model Performance
```
Metrics:
- Loss: 0.0942
- Accuracy: 0.9768
- F1 Score: 0.9768
```

## Setup

Dependencies:
- PyTorch
- Transformers
- Captum
- NumPy
- Pandas
- SciPy
- tqdm


## Analysis Pipeline
1. Load and preprocess text data
2. Extract concept gradients for test set
3. Generate TCAV scores
4. Analyze misclassified samples 
5. Compare concept scores across samples

## References
- Concept Gradient Analysis
- TCAV (Testing with Concept Activation Vectors)
