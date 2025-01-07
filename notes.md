FOR CG

Concept 1 Accuracy: 0.8827
Concept 2 Accuracy: 0.8307
Concept 3 Accuracy: 0.7140
Concept 4 Accuracy: 0.9433
Concept 5 Accuracy: 0.7487

Concept 1 f1-score: 0.8317
Concept 2 f1-score: 0.3172
Concept 3 f1-score: 0.3983
Concept 4 f1-score: 0.9430
Concept 5 f1-score: 0.3968

Label 0       0.82      0.86      0.84      3333
Label 1       0.72      0.52      0.61      1147
Label 2       0.70      0.82      0.76      2898
Label 3       0.88      0.98      0.93      4929
Label 4       0.72      0.53      0.61      1487

Metrics:
 {'eval_loss': 0.24772526323795319, 'eval_f1': 0.7475260036200185, 'eval_accuracy': 0.89616, 'eval_runtime': 10.3991, 'eval_samples_per_second': 961.617, 'eval_steps_per_second': 15.097, 'epoch': 2.983293556085919}

Label distribution across splits:
                  train    dev    test
obscene          117677  12971   32395
sexual_explicit   55183   6029   14910
threat            85340   9471   23370
insult           362424  40337  100487
identity_attack  173962  19375   48181
toxicity          85052   9442   23381

FOR TCAV
Concept 1 Accuracy: 0.832
Concept 2 Accuracy: 0.772
Concept 3 Accuracy: 0.938
Concept 4 Accuracy: 0.818
Concept 5 Accuracy: 0.732

steps: 
1. get the avg cg scores for all the concepts in the test set misclassified sentences .
2. top2 should be insult and obscene
then for each profane word find the number of samples in the training set.

target-model accuracy

{'eval_loss': 0.09423232078552246, 'eval_model_preparation_time': 0.0019, 'eval_accuracy': 0.9768, 'eval_f1': 0.9767981206477725, 'eval_runtime': 4.9787, 'eval_samples_per_second': 2008.555, 'eval_steps_per_second': 31.534}

Confusion Matrix:
 [[4929   71]
 [ 161 4839]]
 