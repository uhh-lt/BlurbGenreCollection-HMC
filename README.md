# BlurbGenreCollection_Classification: Multi-label text classification of writing genres using capsule networks

This page contains the implementation of several neural architectures (CNN, LSTM, Capsule Network) designed for a multi-label text classification task with an underlying hierarchical structure.


The neural networks take as an input a collection of lists of tokens of fixed length, that are referenced by their ID and, as well as a collection of label sets. In case a hierarchy is provided to the program, label correction methods can be applied to create consistent predictions in respect to  the underlying hierarchy. 
Furthermore, the final layers of the neural networks can be pre-initilized with label co-occurrence, as described in [Baker et al.](http://aclweb.org/anthology/W17-2339).

The neural networks and an additional SVM baseline were applied to the BlurbGenreCollection_EN datset, consisting of book blurbs and their respective hierarchically structured writing genres. The datset can be downloaded on the [Language Technology page of the Universität Hamburg](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html).

This page also consists of methods to evaluate the performance of each classifier. An in-depth description and evaluation of the neural networks on the BlurbGenreCollection datset can be found [here](https://www.inf.uni-hamburg.de/en/inst/ab/lt/teaching/theses/completed-theses/2018-ba-aly-blurbs.pdf).


# System Requirement

The system was tested on Debian/Ubuntu Linux with a GTX 1080TI and TITAN X.

# Installation

1. Clone repository: 

  ```
  https://github.com/Raldir/BlurbGenreCollection_Classification.git
  ```
  
2. Install the BlurbGenreCollection-Dataset:

```
cd BlurbGenreCollection_Classification && wget https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection/blurb-genre-collection-en.zip && unzip blurb-genre-collection-en.zip -d datasets
Decompress the .zip
```
  
3. Install project packages:

```
pip install -r code/requirements.txt
```
 
 4. Further packages needed:
 ```
pip install stop-words
python -m spacy download en
python -m spacy download en_core_web_sm
 ```
 
5. Install word embeddings for the English language, e.g.:
```
cd resources && wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec
``` 

# Multi-label Classification

Running the main.py will run the complete Pipeline if in train mode: Loading the data, preprocessing and training the classifier. 
The preprocessed data is stored in the resources folder, to save time in sequential runs. Same applies to the computation of the embedding matrix, which is stored for a fixed sequence length.
However, for the first execution, please run the holdout_train mode (or any mode that uses the dev set) so that the stored preprocessed token collections are generated for each split.  

| Option |  Description | Default|
|--------|-------------|---|
| --mode | Mode, e.g. train on dev using holdout, train on dev and train for testing(train_final) | train_holdout |
| --classifier | Select between CNN, LSTM and capsule | capsule |
| --lang | Datset to be used | EN |

General Settings:

| Option |  Description | Default|
|--------|-------------|---|
| --sequence_length | Maximum sequence imput length of text | 100 |
| --epochs | Number of epochs to train the classifier | 60 |
| --use_statc | Whether the embedding layer should not be trainable | False |
| --use_early_stop |Uses early stopping during training | False |
| --batch_size |Set minibatch size | 32 |
| --learning_rate |The learning rate of the classifier | 0.0005 |
| --learning_decay |Whether to use learning decay, 1 indicates no decay, 0 max.| 1 |
| --iterations |How many classifiers to be trained, only relevant for train_n_models_final | 3 |
| --activation_th |Activation threshold of the final layer | 0.5 |
| --adjust_hierarchy |Postprocessing hierarchy correction | None|
| --correction_th |Threshold for threshold-label correction method | False |


Capsule settings:

| Option |  Description | Default|
|--------|-------------|---|
| --dense_capsule_dim |Dimensionality of capsules on final layer| 16 |
| --n_channels | Number of capsules per feature map | 50 |

LSTM settings:

| Option |  Description | Default|
|--------|-------------|---|
| --lstm_units | Number of units in the lstm | 700 |

CNN settings:

| Option |  Description | Default|
|--------|-------------|---|
| --num_filters | Number of filters for each window size | 500 |


*Example:*  
`python3.5 main.py --mode train_holdout --classifier cnn --lang EN --sequence_length 100 --learning_rate 0.001 --learning_decay 1 
`


For further inquries: 5aly@informatik.uni-hamburg.de

