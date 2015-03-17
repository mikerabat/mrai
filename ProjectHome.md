This library contains classes for easy classifier design. For simple
classifier design check out the Test application - it is basically a unit test
application - and see there which classifieres are available and how
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
  * Ensemble classifiers: Ada Boost, Gentle Boost, Bagging

On top of these classifiers there exists a few image database handling routines
and an 1D, 2D Haar Feature extractor which is based on an integral image approach.

A testing application `TestClassifier.dpr` which shows the usage and performance of these
classifiers on various tasks (e.g. face recognition).

You can for sure also make a donation to the project if you think it is usefull and want to support the developer.

[![](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=ZSLHV2Z36C9K6)