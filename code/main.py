"""
Author: Rami Aly, E-mail: `rami.aly@outlook.com`
"""

from keras.callbacks import ModelCheckpoint
import operator
from data_helpers import load_data, extract_hierarchies, remove_genres_not_level
import numpy as np
import string
import math
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import tensorflow as tf
import itertools
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
import itertools
import os
import traceback
import keras.backend as K
from keras.callbacks import Callback, EarlyStopping
from keras.utils.vis_utils import plot_model
from keras.models import load_model
from keras.preprocessing import sequence
import sys
import argparse
import codecs
import json
import os
import scipy
from networks import create_model_cnn, create_model_lstm, create_model_capsule
import pickle

from keras.callbacks import LearningRateScheduler

#All necessary program arguments are stored here
args = None
#the dataset and vocabulary is stored here
data = {}


def mean_confidence_interval(data, confidence=0.95):
    """
    Calculates mean confidence interval
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def save_scores(results, level):
    """
    Stores the scores into a file, one for each level in hierarchy
    """
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    '../checkpoints','results_' +  args.filename + "_level_"+ str(level) +'.txt')
    out_file = open(filename, 'w')
    metrices = ['f1', 'recall', 'precision', 'accuracy']
    out_file.write("Results on level" + str(level) + '\n')
    print("Level: " + str(level))
    for i in range(len(results[0])):
        mean, lower_confidence, upper_confidence = mean_confidence_interval([element[i] for element in results])
        print("%s: %0.2f \pm %0.2f"%( metrices[i],(mean*100), ((upper_confidence / mean) -1)))
        out_file.write(metrices[i] + ": " + str(mean*100) + " \pm " + str(((upper_confidence / mean) - 1)))
        out_file.write('\n')
    print('\n')
    out_file.close()



class Metrics_eval(Callback):
    """
    Callback to receive score after each epoch of training
    """
    def __init__(self,validation_data):
        self.val_data = validation_data

    def eval_metrics(self):
        #dont use term validation_data, name is reserved
        val_data = self.val_data
        X_test = val_data[0]
        y_test = val_data[1]
        output = self.model.predict(X_test, batch_size = args.batch_size)
        for pred_i in output:
            pred_i[pred_i >=args.activation_th] = 1
            pred_i[pred_i < args.activation_th] = 0
        return [f1_score(y_test, output, average='micro'),f1_score(y_test, output, average='macro'),
         recall_score(y_test, output, average='micro'),precision_score(y_test, output, average='micro'),
          accuracy_score(y_test, output)]

    def on_epoch_end(self, epoch, logs={}):
        f1, f1_macro, recall, precision, acc = self.eval_metrics()
        print("For epoch %d the scores are F1: %0.4f, Recall: %0.2f, Precision: %0.2f, acc: %0.4f, F1_m: %0.4f"%(epoch, f1, recall, precision, acc, f1_macro))
        print((str(precision) + '\n' +  str(recall) + '\n' +
                 str(f1_macro) + '\n' + str(f1) + '\n' + str(acc)).replace(".", ","))


def train(model, save = True, early_stopping = True, validation = True):
    """
    Trains a neural network, can use early_stopping and validationsets
    """

    print("Traning Model...")
    callbacks_list = []
    lr_decay = LearningRateScheduler(schedule=lambda epoch: args.learning_rate * (args.learning_decay ** epoch))
    callbacks_list.append(lr_decay)
    if early_stopping:
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, verbose=0, mode='auto')
        callbacks_list.append(early_stopping)
    if validation:
        metrics_callback = Metrics_eval(validation_data = (data['X_dev'], data['y_dev']))
        callbacks_list.append(metrics_callback)
        model.fit(data['X_train'], data['y_train'], batch_size=args.batch_size, epochs=args.epochs,
         verbose=1, callbacks=callbacks_list, validation_data=(data['X_dev'], data['y_dev'])) # starts training
    else:
        metrics_callback = Metrics_eval(validation_data = (data['X_test'], data['y_test']))
        callbacks_list.append(metrics_callback)
        model.fit(data['X_train'], data['y_train'], batch_size=args.batch_size, epochs=args.epochs, verbose=1,
         callbacks = callbacks_list)


    if save:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print("Saving current Model")
        model.save(os.path.join(save_path, args.filename + '.h5'))



def test(model, data_l, label):
    """
    Tests a neural network on the given data
    """
    global data
    print("Testing Model...")
    print(len(data_l))
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
         '../resources', args.filename + '.output')
    if args.mode == 'evaluate' and os.path.exists(results_path):
        print("Loading model output...")
        output_file = open(results_path, 'rb')
        _,_,_,output,_ = pickle.load(output_file)
    else:
        output = model.predict(data_l, batch_size = args.batch_size)
    binary_output = np.array(output, copy = True)
    #print(binary_output)
    for pred_i in output:
        pred_i[pred_i >=args.activation_th] = 1
        pred_i[pred_i < args.activation_th] = 0

    if args.adjust_hierarchy != 'None' and args.adjust_hierarchy != "threshold":
        output = adjust_hierarchy(output_b = output, language = args.lang,
         mode = args.adjust_hierarchy, max_h = args.level)
    elif args.adjust_hierarchy == "threshold":
        output = adjust_hierarchy_threshold(output = output, output_b = binary_output,
         language = args.lang, max_h = args.level, threshold = args.correction_th)

    results = {}
    if not args.execute_all:
        f1 = f1_score(label, output, average='micro')
        f1_macro = f1_score(label, output, average='macro')
        recall = recall_score(label, output, average='micro')
        precision =  precision_score(label, output, average='micro')
        accuracy = accuracy_score(label, output)
        results[0] = ([f1, recall, precision, accuracy])
    else:
        if args.lang == 'DE':
            levels = [0,1,2]
        elif args.lang =='EN':
            levels = [0,1,2,3]
        else:
            level = [0, 1]
        for level in levels:
            print("Evaluating at level " + str(level) + "...")
            labels_pruned, outputs_pruned = remove_genres_not_level(args.lang,
             label, output, level, exact_level = False)
            f1 = f1_score(labels_pruned, outputs_pruned, average='micro')
            f1_macro = f1_score(labels_pruned, outputs_pruned, average='macro')
            recall = recall_score(labels_pruned, outputs_pruned, average='micro')
            precision =  precision_score(labels_pruned, outputs_pruned, average='micro')
            accuracy = accuracy_score(labels_pruned, outputs_pruned)
            results[level] = ([f1, recall, precision, accuracy])
            print("F1: " + str(f1))
            print("F1_macro: " + str(f1_macro))
            print("Recall: " + str(recall))
            print("Precision: " + str(precision))
            print("Accuracy: " + str(accuracy)+'\n')

    print("F1: " + str(f1))
    print("F1_macro: " + str(f1_macro))
    print("Recall: " + str(recall))
    print("Precision: " + str(precision))
    print("Accuracy: " + str(accuracy))

    return results



def model_cnn(dev, preload = False):
    """
    Creates CNN or loads it if available
    """
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
         '../checkpoints', args.filename + '.h5')
    print(filepath)
    if os.path.isfile(filepath) and preload:
        print ("Loading model...")
        model = load_model(filepath)
        model.summary()
        return model
    else:
        return create_model_cnn(preload, args.embed_dim, args.sequence_length,
         args.num_filters, args.lang,len(data['y_train'][0]), args.use_static,
         args.init_layer, data['vocabulary'], args.learning_rate, dev)


def model_lstm(dev, preload = False):
    """
    Creates LSTM or loads it if available
    """
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
         '../checkpoints', args.filename + '.h5')
    print(filepath)
    if os.path.isfile(filepath) and preload:
        print ("Loading model...")
        model = load_model(filepath)
        model.summary()
        return model
    else:
        return create_model_lstm(preload, args.embed_dim, args.sequence_length,
         args.lstm_units, args.lang, len(data['y_train'][0]),
          args.use_static, args.init_layer, data['vocabulary'], args.learning_rate, dev)


def model_capsule(dev, preload = False):
    """
    Creates capsule networkor loads it if available
    """
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
         '../checkpoints', args.filename + '.h5')
    print(filepath)
    if os.path.isfile(filepath) and preload:
        #model = load_trained_model(filepath, inputs, output)
        print ("Loading model...")
        model = create_model_capsule(preload, args.embed_dim, args.sequence_length,
         args.num_filters, args.lang, len(data['y_train'][0]),
          args.use_static, args.init_layer, data['vocabulary'], args.learning_rate,
          args.dense_capsule_dim, args.n_channels, 3, dev)
        model.load_weights(filepath)
        model.summary()
        return model
    else:
        return create_model_capsule(preload, args.embed_dim, args.sequence_length,
         args.num_filters, args.lang, len(data['y_train'][0]),
          args.use_static, args.init_layer, data['vocabulary'], args.learning_rate,
          args.dense_capsule_dim, args.n_channels, 3, dev)



def main():
    """
    Parses input parameters for networks
    """
    global args
    parser = argparse.ArgumentParser(description="CNN for blurbs")
    parser.add_argument('--mode', type=str, default='train_validation', choices=['train_validation', 'train_test_n_runs', 'train_test'], help="Mode of the system.")
    parser.add_argument('--classifier', type=str, default='cnn', choices=['cnn','lstm', 'capsule'], help="Classifier architecture of the system.")
    parser.add_argument('--lang', type=str, default='EN',  help="Which dataset to use")
    parser.add_argument('--dense_capsule_dim', type=int, default=16, help = 'Capsule dim of dense layer')
    parser.add_argument('--n_channels', type=int, default=50, help = 'number channels of primary capsules')
    parser.add_argument('--batch_size', type=int, default=32, help = 'Set minibatch size')
    parser.add_argument('--level', type=int, default=1, help = "Max Genre Level hierarchy")
    parser.add_argument('--use_static', action='store_true', default=False, help = "Use static embeddings")
    parser.add_argument('--sequence_length', type=int, default=100, help = "Maximum sequence length")
    parser.add_argument('--epochs', type=int, default=60, help = "Number of epochs to run")
    parser.add_argument('--activation_th', type=float, default=0.5, help = "Activation Threshold of output")
    parser.add_argument('--lstm_units', type=int, default=700, help = "Number of units in LSTM")
    parser.add_argument('--num_filters', type=int, default=500, help = "Number of filters in CNN and Capsule")
    parser.add_argument('--adjust_hierarchy', type=str, default='None', choices=['None','semi_transitive', 'transitive', 'restrictive', 'threshold'],
     help = "Postprocessing hierarchy correction")
    parser.add_argument('--correction_th', type=float, default=0.5, help = "Threshold for Hierarchy adjust, in threshold type")
    parser.add_argument('--init_layer', action='store_true', default=False, help = "Init final layer with cooccurence")
    parser.add_argument('--iterations', type=int, default=3, help = "Number of iterations for training")
    parser.add_argument('--embed_dim', type=int, default=300, help = "Embedding dim size")
    parser.add_argument('--use_early_stop', action='store_true', default = False , help = 'Activate early stopping')
    parser.add_argument('--learning_decay', type=float, default = 1., help = 'Use decay in learning, 1 is None')
    parser.add_argument('--learning_rate', type = float, default = 0.0005, help = 'Set learning rate of network')
    parser.add_argument('--execute_all', action='store_true', default = False, help = 'Executes evaluation on every level of hierarchy')
    parser.add_argument('--whitespace_sep', action='store_true', default = False, help = 'Uses whitespace seperation instead of spacy')
    parser.add_argument('--filter_low_freq', action='store_true', default = False, help = 'Filter low frequency words from dataset')
    #0.001

    args = parser.parse_args()
    import json
    params = vars(args)
    print(json.dumps(params, indent = 2))
    run()



def run():
    """
    Execution pipeline for each mode
    """
    classifier = args.classifier

    #used for training the model on train and dev, executes only once, simpliest version
    if args.mode =='train_test':
        init_data(dev = False)
        model = create_model(dev = False, preload = False)
        train(model,  early_stopping = args.use_early_stop, validation = False)
        test(model, data_l = data['X_test'], label = data['y_test'])

    #uses holdout to train n models. This mode used for parameter optimization
    elif args.mode == 'train_validation':
        init_data(dev = True)
        results_dict = {}
        for i in range(args.iterations):
            model = create_model(dev = True, preload = False)
            train(model, early_stopping = args.use_early_stop, validation = True, save = False)

            result_hierarchies = test(model, data_l = data['X_dev'], label = data['y_dev'])
            for results in result_hierarchies:
                if results in results_dict:
                    results_dict[results].append(result_hierarchies[results])
                else:
                    results_dict[results] = [result_hierarchies[results]]
        for hierarchy in results_dict:
            save_scores(results_dict[hierarchy], hierarchy)

    elif args.mode == 'train_test_n_runs':
        init_data(dev=False)
        results_dict = {}
        results = []
        for i in range(args.iterations):
            model = create_model(dev = False, preload = False)
            train(model, early_stopping = False, validation = False)
            result_hierarchies = test(model, data_l = data['X_test'], label = data['y_test'])
            for results in result_hierarchies:
                if results in results_dict:
                    results_dict[results].append(result_hierarchies[results])
                else:
                    results_dict[results] = [result_hierarchies[results]]
        for hierarchy in results_dict:
            save_scores(results_dict[hierarchy], hierarchy)

    print(args.filename)

    K.clear_session()


def create_model(dev = False, preload = True):
    """
    General method to create model based on user arguments
    """
    general_name = ("__batchSize_" + str(args.batch_size) + "__epochs_" + str(args.epochs)
    + "__sequenceLen_" + str(args.sequence_length)  + "__activThresh_" + str(args.activation_th) + "__initLayer_"
    + str(args.init_layer) + "__adjustHier_" + str(args.adjust_hierarchy) +  "__correctionTH_"
    + str(args.correction_th) + "__learningRate_" + str(args.learning_rate) + "__decay_"
    + str(args.learning_decay) + "__lang_" + args.lang)
    if args.classifier == 'lstm':
        args.filename = ('lstm__lstmUnits_' + str(args.lstm_units) + general_name)
        return model_lstm(dev, preload)
    elif args.classifier == 'cnn':
        args.filename = ('cnn__filters_' + str(args.num_filters) + general_name)
        return model_cnn(dev, preload)
    elif args.classifier == 'capsule':
        args.filename = ('capsule__filters_' + str(args.num_filters) + general_name)
        return model_capsule(dev, preload)
    print(args.filename)


def init_data(dev, outlier = False):
    """
    Initilizes the data(splits) and vocabulary
    """
    global data
    use_spacy = not args.whitespace_sep
    use_low_freq = not args.filter_low_freq
    if dev:
        X_train, y_train, X_dev, y_dev, X_test, y_test, vocabulary, vocabulary_inv =load_data(spacy = use_spacy, lowfreq = use_low_freq,
         max_sequence_length =  args.sequence_length, type = args.lang, level = args.level, dev = dev)
        data['X_dev'] = X_dev
        data['y_dev'] = y_dev

    else:
        X_train, y_train, X_test, y_test, vocabulary, vocabulary_inv = load_data(spacy = use_spacy, lowfreq = use_low_freq,
         max_sequence_length =  args.sequence_length, type = args.lang, level = args.level, dev = dev)

    data['X_train'] = X_train
    data['y_train'] = y_train
    data['X_test'] = X_test
    data['y_test'] = y_test
    data['vocabulary'] = vocabulary
    data['vocabulary_inv'] = vocabulary_inv

if __name__ == '__main__':
    main()
