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
* C4.5 Decission trees.
* Ensemble classifiers: AdaBoost, Gentle Boost, Bagging

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
https://code.google.com/p/mrmath/
https://code.google.com/p/mrimgutil/

First download both of these libraries and compile the included dpk files.
Also add the directories to the library (and or) search paths!

Platforms:

This library was built mainly for the windows platform and is 
compatible with Delphi2007 and later versions. 

Developer:

Rabatscher Michael contact via www.mrsoft.com