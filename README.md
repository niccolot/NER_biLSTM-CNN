Implementation of the named entity recognition (NER) neural network based on the article https://arxiv.org/abs/1511.08308, featuring in particular 
a CNN embedding layer for character-level embedding.

# Contents

- **data** contains the .txt file from the CoNLL-2003 train dataset, every dataset formatted as 

~~~
EU NNP B-NP B-ORG
rejects VBZ B-VP O
German JJ B-NP B-MISC
call NN I-NP O
to TO B-VP O
boycott VB I-VP O
British JJ B-NP B-MISC
lamb NN I-NP O
. . O O
~~~

i.e. each word in a line and sentences separated by a new line (\n), will be ok with the preprocessing functions

- **preprocessing.py** contains preprocessing utility functions used from the read of the dataset and embedding .txt file to the output of a suitable dataset
- **architecture.py** contains the custom classes and a function returning the whole model
- **train.py** is a script for the training

# Handpicked hyperparameters

Concerning the padding, the max length for a sentence has been fixed to 50 words and the max character length has been fixed to 15. 
Both of thees after plotting the distribution of sentences and words length:

<img src= "figs/senteces_length_hist.png" height='350' width='450'> <img src= "figs/words_length.png" height='350' width='450'>

it is clear that most sentences have at most 50 words and most words have at most 15 characters.
