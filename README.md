## Objectives

The objectives of this third assignment are to:

1.  Write a linear regression program using gradient descent
2.  Write linear classifiers using the perceptron algorithm and logistic regression.
3.  Experiment variations of the algorithms.
4.  Evaluate your classifier.
5.  Present your code and results in a short dissertation.

## Overview

The gradient descent is a basic technique to estimate linear discriminant functions. You will first use the gradient descent method to implement linear regression. You will then program the perceptron algorithm. Finally, you will improve the threshold function with the logistic curve. You will try various configurations and study their influence on the learning speed and accuracy.

As programming language, you will use Python (strongly preferred) or, possibly, Java.

## Linear Regression

Implement the gradient descent method as explained in pages 719--720 in Russell-Norvig and in the slides to compute regression lines. You will implement stochastic and batch versions of the algorithm.

You will test your program on two data sets corresponding to letter counts in the 15 chapters of the French and English versions of _Salammbô_, where the first column is the total count of characters and the second one, the count of A's: [[French](http://fileadmin.cs.lth.se/cs/Education/EDA132/Labs/ML/salammbo_a_fr.plot)] [[English](http://fileadmin.cs.lth.se/cs/Education/EDA132/Labs/ML/salammbo_a_en.plot)]

Before you start the computation, scale the data so that they fit in the range [0, 1] on the x and y axes. Try different values for the learning rate.

Visualize the points as well as the regression lines you obtain using matplotlib or another similar program.

## The Perceptron

You will use the same data set as for linear regression. You will encode the classes and the features using the LIBSVM format, also called SVMLight. This format is a standard way to encode data sets and you can find a description [here](https://github.com/cjlin1/libsvm/blob/master/README). You can also read details on the [sparse data format](http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#/Q3:_Data_preparation) as a complement. The complete LIBSVM program is available from this page: [http://www.csie.ntu.edu.tw/~cjlin/libsvm/](http://www.csie.ntu.edu.tw/~cjlin/libsvm/). You do not need this program in the assignment. You just need to read the description of the data format.

Write a reader function for the LIBSVM format and scale the data in your set. You can write a simplified reader that assumes that all the attributes, including zeros, will have an index, i.e. ignore the sparse format.

Write the perceptron program as explained in pages 723--725 in Russell-Norvig and in the slides and run it on your data set.

As a stop criterion, you will use the number of misclassified examples.

Report the results of the classification and the parameters you have used.

Evaluate your perceptron using the leave-one-out cross validation method.

## Logistic Regression

From your perceptron program, implement logistic regression. You can either follow the description from the textbook, S. Russell and R. Norvig, _Artificial Intelligence_, 2010, pages 725--727, or the slides. You can either implement the stochastic or the batch version of the algorithm, or both versions. Run the resulting program on your data set.

Evaluate your logistic regression using the leave-one-out cross validation method.

## Remarks

### Deadline

Use Overleaf to write your report and send the link before 23.59 on March 1st, 2019\. The link should be e-mailed to tai @ cs.lth.se with the subject line Assignment X by username1/username2.

### Problems

In case of problems, send an e-mail to Pierre.Nugues@cs.lth.se.

### Report

The assignment must be documented in the report, which should contain the following:

*   The name of the author, the title of the assignment, and any relevant information on the front page.
*   A presentation of the assignment and the possible improvements you would have brought.
*   A presentation of your implementation and how to run the executable.
*   A print-out of the example set(s) and the resulting weight vectors.
*   Comments on the results you have achieved.

Please, typeset and format your report consistently. You must use Latex.

You may have a look to the code in the textbook code repository (or any other implementations), but the code you hand in must be your work. You do not need to provide all or any code printout in the report -- the code is available in your solution directory anyway -- but only comments on its interesting parts.

## Programming Language and Environment

You can use one of these languages: Python3 (preferred) or Java and your own computer to develop your program. No other programming language is allowed.

Your final program must be available and runnable on the LINUX computers at the *.student.lth.se address (e.g. login.student.lth.se). Remember to make your programs and all the directories in their path read and execute accessible to 'others' (chmod 705 filename). Remember also to quote where does your solution reside and how should it be run (kind of "User's Guide"). You can also upload it in a subfolder in your Overleaf folder.

The resulting programs should remain in your directory until you have been notified of the result, e.g. on the notice board and/or web or by e-mail. You may expect that your report and implementation will be examined within two weeks. If your report or implementation is unsatisfactory you will be given one chance to make the corrections and then to hand it in within a week after you have been notified (on the notice board and/or web or by e-mail).
