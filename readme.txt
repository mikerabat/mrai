// ###################################################################
// #### This file is part of the mathematics library project, and is
// #### offered under the licence agreement described on
// #### http://www.mrsoft.org/
// ####
// #### Copyright:(c) 2014, Michael R. . All rights reserved.
// ###################################################################

This library contains classes for easy classifier design. For simple
classifier design check out the TestApp - basically this is a unit test
application -and see there which classifieres are available and how
to use them. It's recommended to have base knowledge about the underlying
algorithms. Quite a few algorithms have been designed with 
robustness in mind (robustness in terms of outlying pixels or occluded areas).

The package includes:

* Standard Fisher LDA classifier
* Robust (and Fast Robust) version of this classifier
* Incremental (and Robust) Fisher LDA classifier learning.
* Support Vector Machines (least squares and lagrangian learning)
* Naive Bayes
* Simple Decission stumps
* Radial basis function
* C4.5 Decission trees.
* K-means
* Ensemble classifiers: AdaBoost, Gentle Boost, Bagging
* Simple feed forward Neural Nets

On top of these classifiers there exists a few image database handling routines
and an 1D, 2D Haar Feature extractor which is based on an integral image approach.

A testing application TestClassifier.dpr which shows the usage and performance of these
classifiers on various tasks (e.g. face recognition).

Installation:

This library is an extension of the mrMath and mrImgUtils libraries and 
therefore depending on it!

You can check out the libraries from 
http://www.mrsoft.org/

or
https://github.com/mikerabat/mrmath
https://github.com/mikerabat/mrimgutil

First download both of these libraries and compile the included dpk files.
Also add the directories to the library (and or) search paths!

Platforms:

This library was built mainly for the windows platform and is 
compatible with Delphi2007 and later versions. 

Developer:

Rabatscher Michael contact via www.mrsoft.org


changelog:

Date 26.05.2021:
* Various bugfixes for SVM.
* Fixed a problem in the threaded decission stump learner where the threshold calculation was off by one.
* simplified decission stump learner code

Date 27.4.2020:
* speed up of the c4.5 implementation (thanks for the valuable input of N. Peladeau)
* Multithreaded c4.5
* Added some new features for c4.5: Stopping if a tree part has less then a certain minimum examples.

Date: 12.06.2017:
* Implemented new idea for weighted neural network learning (weight multiplication on backprop step)

Date: 16.03.2017:
* started change log
* Added quite a few confidence calculations: svm, lda, neural network, rbf, naive 
  bayes, kmeans