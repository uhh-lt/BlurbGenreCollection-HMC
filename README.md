# BlurbGenreCollection_Classification: Multi-label text classifcation of writing genres using capsule networks

This page contains the implementation of several neural architectures (CNN, LSTM, Capsule Network) designed for a multi-label text classification task with an underlying hierarchical structure.


The neural networks take as an input a collection of lists of tokens referenced by their id and fixed length, as well as the collection of the label sets. In case a hierarchy is provided to the program, label corerction methods can be applied to create consistent predictions in respect to  the underlying hierarchy. 
Furthermore, the final layers of the neural networks can be pre-initilized with label-cooccurence, as described in [Baker et. all](http://aclweb.org/anthology/W17-2339).

The neural networks and an additional SVM baseline were applied to the BlurbGenreCollection_EN datset, consisting of book blurbs and their respective hierarchically structured writing genres. The datset can be downloaded on the Language Technology page of the Universitaet Hamburg https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html.

This page also consists of methods to evaluate the performance of each classifier. An in-depth description and evaluation of the neural networks on the BlurbGenreCollection datset can be found in: https://www.inf.uni-hamburg.de/en/inst/ab/lt/teaching/theses/completed-theses/2018-ba-aly-blurbs.pdf.


# System Requirement

The system was tested on Debian/Ubuntu Linux with a GTX 1080TI and TITAN X.

# Installation




