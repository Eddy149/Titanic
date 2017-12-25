# Titanic
This is my attempt to solve the Kaggle Titanic challenge using octave script. 
See: https://www.kaggle.com/c/titanic

I start by predicting survival with logistic regression, add regularization, then train a neural network with 2 hidden layers of size 5 each. Hyperparameters (lambda, thresholds)  are chosen from a hold-out cross validation set that represents roughly 30% of the original training dataset.

This draws on some code from Andrew Ng's excellent Machine Learning Coursera class.
https://www.coursera.org/learn/machine-learning/home/welcome

Possible avenues of further development: 
1) normalize features
2) draw learning curves
3) automate choice of lambda with an optimization algorithm
4) experiment with neural networks of different size. 
5) modify from train/CV approach to a n-fold approach to selecting hyperparameters


