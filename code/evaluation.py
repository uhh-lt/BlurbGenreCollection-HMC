import matplotlib
import scipy.stats
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import sys
from data_helpers import ml, remove_genres_not_level, adjust_hierarchy, adjust_hierarchy_threshold, load_data
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import operator
import itertools
from loader import read_all_genres
import math
from keras.models import load_model
from keras import backend as K
from networks import create_model_capsule
import pickle
args = None


def analysis(data, labels, output, binary_output, args_o, data_all):
    """
    executes complete analysis pipeline
    """
    global args
    args = args_o

    output_labels = ml.inverse_transform(output)
    actual_labels = ml.inverse_transform(labels)

    results_log(output_labels, binary_output, actual_labels, data_all['vocabulary_inv'], data)
    evaluate_label_correction(output, binary_output, labels)
    evaluate_frequency_performance(labels, output, actual_labels, output_labels, args.lang)
    evaluate_level_performance(labels, output)
    #create_confusion_matrix(actual_labels, output_labels, display_top_n = 25, reverse = False)
    create_confusion_matrix(actual_labels, output_labels, display_top_n = 75, reverse = True)



def results_log(output_labels, binary_output, actual_labels, vocabulary_inv,data):
    """
    For each blurb, stores prediction, actual labels and confidence into file
    """
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
     '../checkpoints','predictions_' +  args.filename + '.txt')

    with open(filename, 'w') as f:
        for i in range(0, len(output_labels)):
            text = []
            for index in data[i]:
                f.write(vocabulary_inv[index] + " ")
            f.write('\n')
            f.write('Binary labels: ')
            for label_b in binary_output[i]:
                #print(label_b)
                f.write(str(label_b) + " ")
            f.write('\n')
            f.write('System prediction: ')
            for label_s in output_labels[i]:
                f.write(str(label_s) + " ")
            f.write('\n')
            f.write('Actual prediction: ')
            for label_a in actual_labels[i]:
                f.write(str(label_a) + " ")
            f.write('\n')
            f.write('\n')
        f.close()



def evaluate_label_correction(output_o, binary_output, labels):
    """
    Executes all label correction methods on output of nn
    """
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../resources', args.filename + '.label_correction')
    file  = open(filepath, 'w')

    threshold = 0.2
    output = adjust_hierarchy_threshold(output = binary_output, output_b = output_o, language = args.lang, max_h = args.level, threshold = threshold)
    f1 = f1_score(labels, output, average='micro')
    recall = recall_score(labels, output, average='micro')
    precision =  precision_score(labels, output, average='micro')
    accuracy = accuracy_score(labels, output)
    file.write("threshhold_correction_" + str(threshold) + ':\n')
    file.write("F1: " + str(f1) + '\n')
    file.write("Recall:" + str(recall) + '\n')
    file.write("Precision: " + str(precision) + '\n')
    file.write("Accuracy: " + str(accuracy) + '\n \n')
    print([f1, recall, precision, accuracy])
    methods = ["semi_transitive", "transitive", "restrictive"]
    for method in methods:
            output = adjust_hierarchy(output_b = output_o, language = args.lang, mode = method, max_h = args.level)
            f1 = f1_score(labels, output, average='micro')
            recall = recall_score(labels, output, average='micro')
            precision =  precision_score(labels, output, average='micro')
            accuracy = accuracy_score(labels, output)
            print([f1, recall, precision, accuracy])
            file.write(method + ':\n')
            file.write("F1: " + str(f1) + '\n')
            file.write("Recall:" + str(recall) + '\n')
            file.write("Precision: " + str(precision) + '\n')
            file.write("Accuracy: " + str(accuracy) + '\n \n')



def evaluate_frequency_performance(label, output, actual_labels, output_labels, lang, plot = True):
    """
    creates dictionary: Score for each label combination in trainingset
    Can additionally plot that with regressioncurve
    """
    _, occurences = read_all_genres(lang)
    predicted_results = []
    print(len(occurences))
    for occurence in occurences:
        predictions = []
        actual = []
        curr_gen  = None
        for i,predict in enumerate(output):
            #print(actual_labels[i], occurence[0])
            if set(actual_labels[i]) == occurence[0]:
                curr_gen = label[i]
                predictions.append(predict)

        for i in range(len(predictions)):
            actual.append(occurence[0])
        actual =  ml.transform(actual)
        score = f1_score(np.array(actual), np.array(predictions), average='micro')
        predicted_results.append(score)
    print(len(predicted_results))
    occurence_freq = [occurence[1] for occurence in occurences]
    occurence_result = zip(occurence_freq, predicted_results)
    occurence_result = sorted(occurence_result, key =operator.itemgetter(0))
    occurence_freq  = [math.log10(occurence[0]) for occurence in occurence_result]
    scores = [occurence[1] for occurence in occurence_result]
    if plot:
        plt.grid(True)
        fit = np.polyfit(occurence_freq, scores,1)
        fit_fn = np.poly1d(fit)
        plt.plot(occurence_freq, scores, 'yo', occurence_freq, fit_fn(occurence_freq), 'k')
        plt.ylabel("score")
        plt.xlabel("log10 #occurence")
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../resources', args.filename + 'occurence_score.png'))
    return occurence_result



def evaluate_level_performance(label, output):
    """
    Computes scores for each hierarchy level seperatly
    """
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../resources', args.filename + '.level_scores')
    file  = open(filepath, 'w')
    results = {}
    if args.lang == 'DE':
        levels = [0,1,2]
    elif args.lang =='EN':
        levels = [0,1,2,3]

    for level in levels:
        labels_pruned, outputs_pruned = remove_genres_not_level(args.lang,
         label, output, level, exact_level = True)

        f1 = f1_score(labels_pruned, outputs_pruned, average='micro')
        recall = recall_score(labels_pruned, outputs_pruned, average='micro')
        precision =  precision_score(labels_pruned, outputs_pruned, average='micro')
        accuracy = accuracy_score(labels_pruned, outputs_pruned)
        file.write(str(level) + ':\n')
        file.write("F1: " + str(f1) + '\n')
        file.write("Recall:" + str(recall) + '\n')
        file.write("Precision: " + str(precision) + '\n')
        file.write("Accuracy: " + str(accuracy) + '\n \n')
        results[level] = ([f1, recall, precision, accuracy])
        print(results[level])



def create_confusion_matrix(actual_labels, output_labels, display_top_n = 10, reverse = True):
    """
    Creates confusion matrix for top n co_occurences
    """
    print("Creating Confusion-matrix...")
    frequencies = {}
    for label in output_labels:
        label_string = ';'.join(label)
        if label_string in frequencies:
            frequencies[label_string]+=1
        else:
            frequencies[label_string] =1

    if reverse:
        occurences = sorted(frequencies.items(), key=operator.itemgetter(1), reverse = True)
    else:
        occurences = sorted(frequencies.items(), key=operator.itemgetter(1), reverse = False)
    total_occurences = len(occurences)
    occurences_s = [occurence[0] for occurence in occurences][:display_top_n] + ['Other']
    occurences_s = [element if element != '' else 'None' for element in occurences_s]
    occurences = [occurence[0].split(";") for occurence in occurences][:display_top_n]
    if [''] in occurences:
        occurences[occurences.index([''])] = []
    output_labels_id = [';'.join(label) if list(label) in occurences else
     'Other' for label in output_labels]
    actual_labels_id = [';'.join(label) if list(label) in occurences else
     'Other' for label in actual_labels]
    conf_matrix = confusion_matrix(actual_labels_id, output_labels_id, occurences_s)
    plot_confusion_matrix(conf_matrix, classes=occurences_s,
     title='Confusion matrix. Most ' + str(display_top_n) + " frequent label-combinations out of " +  str(total_occurences) + ' label co-occurences', reverse = reverse)



def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, reverse = True):

    """
    Plots confusion matrix
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    if not args.level == 0:
        plt.figure(figsize=(40, 40))
        #plt.figure(figsize=(20,20))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha = 'right')
    plt.yticks(tick_marks, classes)

    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
    '../resources', args.filename + "_" + str(reverse) + "_" + str(args.level) + '.pdf'))



def main(lang):
    """
    Plots the label co-occurences for all models togehter in one graph
    @require .output file for every reference model mentioned
    """
    if lang == 'EN':
        path_capsule =os.path.join(os.path.dirname(os.path.abspath(__file__)), '../resources', 'capsule__filters_500__batchSize_32__epochs_4__sequenceLen_100__activThresh_0.5__initLayer_False__adjustHier_None__correctionTH_0.5__learningRate_0.001__decay_0.4__lang_EN')
        path_cnn = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../resources', 'cnn__filters_1500__batchSize_32__epochs_30__sequenceLen_100__activThresh_0.5__initLayer_False__adjustHier_None__correctionTH_0.5__learningRate_0.0005__decay_0.9__lang_EN')
        path_lstm = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../resources',
        'lstm__lstmUnits_1500__batchSize_32__epochs_15__sequenceLen_100__activThresh_0.5__initLayer_False__adjustHier_None__correctionTH_0.5__learningRate_0.0005__decay_1.0__lang_EN')
    elif lang == 'DE':
        path_cnn =  os.path.join(os.path.dirname(os.path.abspath(__file__)),'../resources', 'cnn__filters_1500__batchSize_32__epochs_30__sequenceLen_100__activThresh_0.5__initLayer_False__adjustHier_None__correctionTH_0.5__learningRate_0.0005__decay_1.0__lang_DE')
        path_lstm =  os.path.join(os.path.dirname(os.path.abspath(__file__)),'../resources', 'lstm__lstmUnits_1500__batchSize_32__epochs_25__sequenceLen_100__activThresh_0.5__initLayer_False__adjustHier_None__correctionTH_0.5__learningRate_0.001__decay_1.0__lang_DE')
        path_capsule = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../resources','capsule__filters_502__batchSize_32__epochs_7__sequenceLen_100__activThresh_0.5__initLayer_False__adjustHier_None__correctionTH_0.5__learningRate_0.001__decay_0.92__lang_DE')
    paths = [path_cnn, path_lstm, path_capsule]
    results = []
    for i,path in enumerate(paths):
        result_path = path + '.output'
        if os.path.isfile(result_path):
            result_file = open(result_path, 'rb')
            results.append(pickle.load(result_file))
        else:
            print(path, "Model does not exist")
            return
    load_data(spacy = True,max_sequence_length =  100, type = lang, level = 3, dev = False)
    evaluate_frequency_performance_all(lang, results[0], results[1], results[2])


def evaluate_frequency_performance_all(lang, data_cnn, data_lstm, data_capsule):
    "Plot frequency performances of all models on one graph and their respective regression line"
    _, label_cnn, output_cnn, _, _ = data_cnn
    _, label_lstm, output_lstm, _, _ = data_lstm
    _, label_capsule, output_capsule, _, _ = data_capsule
    actual_label_cnn = ml.inverse_transform(label_cnn)
    actual_label_lstm = ml.inverse_transform(label_lstm)
    actual_label_capsule = ml.inverse_transform(label_capsule)
    output_labels_cnn = ml.inverse_transform(output_cnn)
    output_labels_lstm = ml.inverse_transform(output_lstm)
    output_labels_capsule = ml.inverse_transform(output_capsule)
    occurence_result_cnn = evaluate_frequency_performance(label_cnn, output_cnn, actual_label_cnn, output_labels_cnn, lang, plot = False)
    occurence_result_lstm = evaluate_frequency_performance(label_cnn, output_lstm, actual_label_lstm, output_labels_lstm, lang,  plot = False)
    occurence_result_capsule = evaluate_frequency_performance(label_capsule, output_capsule, actual_label_capsule, output_labels_capsule, lang, plot = False)

    occurence_freq_cnn  = [math.log10(occurence[0]) for occurence in occurence_result_cnn]
    scores_cnn = [occurence[1] for occurence in occurence_result_cnn]
    occurence_freq_lstm = [math.log10(occurence[0]) for occurence in occurence_result_lstm]
    scores_lstm = [occurence[1] for occurence in occurence_result_lstm]
    occurence_freq_capsule  = [math.log10(occurence[0]) for occurence in occurence_result_capsule]
    scores_capsule = [occurence[1] for occurence in occurence_result_capsule]

    fit_cnn = np.polyfit(occurence_freq_cnn, scores_cnn,1)
    fit_fn_cnn = np.poly1d(fit_cnn)
    fit_lstm = np.polyfit(occurence_freq_lstm, scores_lstm,1)
    fit_fn_lstm = np.poly1d(fit_lstm)
    fit_capsule = np.polyfit(occurence_freq_capsule, scores_capsule,1)
    fit_fn_capsule = np.poly1d(fit_capsule)


    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row', figsize= (10,10))
    f.text(0.5, 0.04,"(Log10) Label Occurence", ha = 'center')
    f.text(0.04, 0.5,"[F1]-Score", ha = 'center', rotation = 'vertical')
    ax1.set_title("cnn")
    ax2.set_title('lstm')
    ax3.set_title('capsule')
    ax4.set_title('regression line comparison')
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)

    ax1.plot(occurence_freq_cnn, scores_cnn, 'yo', occurence_freq_cnn, fit_fn_cnn(occurence_freq_cnn), 'b')
    ax2.plot(occurence_freq_lstm, scores_lstm, 'yo', occurence_freq_lstm, fit_fn_lstm(occurence_freq_lstm), 'r')
    ax3.plot(occurence_freq_capsule, scores_capsule, 'yo', occurence_freq_capsule, fit_fn_capsule(occurence_freq_capsule), 'g')
    ax4.plot(occurence_freq_capsule, fit_fn_capsule(occurence_freq_capsule),'g', label = 'capsule')
    ax4.plot(occurence_freq_lstm, fit_fn_lstm(occurence_freq_lstm), 'r', label = 'lstm')
    ax4.plot(occurence_freq_cnn, fit_fn_cnn(occurence_freq_cnn), 'b', label = 'cnn')
    ax4.legend(loc = 'upper left')
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../resources','all_occurence_score_' + lang + '.png'))


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        language = sys.argv[1]
    main(language)
