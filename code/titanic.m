%{

This is my attempt to solve the Kaggle Titanic challenge using octave script

I start by predicting survival with logistic regression, add regularization, then 
train a neural network

%}

  
clear ; close all ; clc ; 

cd 'C:\Users\MichaelEddy\Box Sync\Library\Machine Learning\Titanic\data';
addpath("../code")


%[Survived, PassengerId, Pclass, Sex(female=1), Age, SibSp, Parch, Fare, Cabin, Embarked (S=1, C=2 , Q=3] 
data = csvread('train_mod.csv');
data_test = csvread('test_mod.csv');
dataTr = data(1:630,:);
dataCV = data(631:end,:);

% leave out the passenger ID (column 2) 
y_cv = dataCV(:,1);
X_cv = dataCV(:,3:end);

y = dataTr(:,1);
X = dataTr(:,3:end);

[m, n] = size(X);
[m_cv, n_cv] = size(X_cv);

%{
% PLOT DATA  
% just plotting the data on fare and age.   
plotData(X(:,[3,6]),y)
hold on;
xlabel('age')
ylabel('fare')
legend('survived', 'did not survive')
hold off;
%}

%% ============= Part 1: Initialize =============

X = [ones(m, 1) X];
X_cv = [ones(m_cv, 1) X_cv];
initial_theta = zeros(n + 1, 1);
lambda = 2;

%% ============= Part 2: Estimate thetas  =============
%  In this exercise, you will use a built-in function (fminunc) to find the
%  optimal parameters theta.

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);


%% ============= Part 3: Picking lambda using CV =============
%{
%% pick lambda from CV set %%
%y_hat = X*theta;

[lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, X_cv, y_cv);

close all;
%plot(lambda_vec, error_train, lambda_vec, error_val);
plot(lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end
% optimal lambda is 2 (by visual inspection)
%}



%% ============= Part 4: pick a threshold value using CV set=============
y_hat_cv = sigmoid(X_cv*theta);


[threshold, acc] = fminsearch(@(t)(-accuracy_titanic(t,y_hat_cv,y_cv)), 0.5);

threshold
acc = -acc

%% ============= Part 5: run on test set and output submission=============
data_test = csvread('test_mod.csv');
[m_test, n_test] = size(data_test);
X_test = [ones(m_test, 1) data_test(:,2:end)];

y_hat_test = sigmoid(X_test*theta);

y_hat_test = y_hat_test>threshold;

csvwrite("submission_v1.csv", [data_test(:,1) y_hat_test]);


%% ============= Part 6: initialize Neural Network =============

% we're going to design a neural network with an input layer of size 7,
% two hidden layers of size 5 and and an output layer of size 1.

clear initial_theta;

%for this part we drop the initial bias term we had added before
X = X(:, 2:end);
X_cv = X_cv(:, 2:end);
X_test = X_test(:, 2:end);


input_layer_size  = 7;  
hidden_layer1_size = 5;   
hidden_layer2_size = 5;   
output_layer_size = 1; 


initial_theta1 = randInitializeWeights(input_layer_size,hidden_layer1_size);
initial_theta2 = randInitializeWeights(hidden_layer1_size,hidden_layer2_size);
initial_theta3 = randInitializeWeights(hidden_layer2_size,1);

initial_nn_params = [initial_theta1(:) ; initial_theta2(:); initial_theta3(:)];


% make sure cost function is working alright

lambdaNN = .001 ;

%% ============= Part 7: Train Neural Network =============
%now let's do this!
options = optimset('MaxIter', 300);
costFn = @(p) nnCostFunction(p,input_layer_size,hidden_layer1_size,hidden_layer2_size, ...
                  output_layer_size, X,y,lambdaNN) ;

%Now optimize! (fmincg is is a slight modification of fmincg that displays progress)
[ nn_params, cost] = fmincg(costFn, initial_nn_params, options);

%repeated a couple times to pick lambda based on cost estimated on the CV set. (i picked .01)
Jnn = nnCostFunction(nn_params,input_layer_size,hidden_layer1_size,hidden_layer2_size, ...
                  output_layer_size, X_cv,y_cv, lambdaNN )


% unroll nn_params
Theta1 = reshape(nn_params(1:hidden_layer1_size * (input_layer_size + 1)), ...
                 hidden_layer1_size, (input_layer_size + 1));

t= ((hidden_layer1_size * (input_layer_size + 1))) + ... 
  ((hidden_layer1_size+1)*hidden_layer2_size);
Theta2 = reshape(nn_params((1 + (hidden_layer1_size * (input_layer_size + 1))):t), ...
                 hidden_layer2_size, (hidden_layer1_size + 1));
                
t = ((input_layer_size+1)*hidden_layer1_size) + ((hidden_layer1_size+1)*hidden_layer2_size);
Theta3 = reshape(nn_params((1 + t):end), ...
                 output_layer_size, (hidden_layer2_size + 1));

%% ============= Part 8: Predict in CV set and test set =============
%predict & check accuracy with a given threshold (0.5)
y_hatNN = predict_titanicNN(X,Theta1,Theta2,Theta3);
y_hatNN_cv = predict_titanicNN(X_cv,Theta1,Theta2,Theta3);
accuracy_titanic(.5, y_hatNN_cv, y_cv);

%now optimize threshold, and spit out accuracy & threshold
[thresholdNN, accNN] = fminsearch(@(t)(-accuracy_titanic(t,y_hatNN_cv,y_cv)), 0.5)


%now run on test set and output!
y_hat_test = predict_titanicNN(X_test,Theta1,Theta2,Theta3);
y_hat_test = y_hat_test>threshold;

csvwrite("submission_NN.csv", [data_test(:,1) y_hat_test]);

