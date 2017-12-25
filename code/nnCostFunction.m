function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer1_size, ...
                                   hidden_layer2_size, ...
                                   output_layer_size, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, output_layer_size, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer1_size * (input_layer_size + 1)), ...
                 hidden_layer1_size, (input_layer_size + 1));

t= ((hidden_layer1_size * (input_layer_size + 1))) + ... 
  ((hidden_layer1_size+1)*hidden_layer2_size);
                 
Theta2 = reshape(nn_params((1 + (hidden_layer1_size * (input_layer_size + 1))):t), ...
                 hidden_layer2_size, (hidden_layer1_size + 1));
                
               
t = ((input_layer_size+1)*hidden_layer1_size) + ((hidden_layer1_size+1)*hidden_layer2_size);
              
Theta3 = reshape(nn_params((1 + t):end), ...
                 output_layer_size, (hidden_layer2_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

K = size(Theta2,1);



% FEED FORWARD

a1 = [ones(m,1), X];
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1), 1), a2]; 

z3 = a2 * Theta2';
a3 = sigmoid(z3);
a3 = [ones(size(a2,1), 1), a3]; 

z4 = a3 * Theta3';
h = sigmoid(z4);



% COMPUTE COST
% compute penalty
p = sum(sum(Theta1(:, 2:end).^2, 2))+sum(sum(Theta2(:, 2:end).^2, 2))+sum(sum(Theta3(:, 2:end).^2, 2));

% compute regularized cost
J = sum(sum((-y).*log(h) - (1-y).*log(1-h), 2))/m + lambda*p/(2*m);



%--------------- BACK PROPOGATION for partial derivatives---------------%


Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));


% calculate sigmas
sigma4 = h.-y;
sigma3 = (sigma4*Theta3).*sigmoidGradient([ones(size(z3, 1), 1) z3]);
sigma3 = sigma3(:, 2:end);
sigma2 = (sigma3*Theta2).*sigmoidGradient([ones(size(z2, 1), 1) z2]);
sigma2 = sigma2(:, 2:end);

% accumulate gradients
delta_1 = (sigma2'*a1);
delta_2 = (sigma3'*a2);
delta_3 = (sigma4'*a3);

%calculate regularized gradient
p1 = (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
p2 = (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
p3 = (lambda/m)*[zeros(size(Theta3, 1), 1) Theta3(:, 2:end)];

Theta1_grad = delta_1./m + p1;
Theta2_grad = delta_2./m + p2;
Theta3_grad = delta_3./m + p3;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:); Theta3_grad(:)];



end

