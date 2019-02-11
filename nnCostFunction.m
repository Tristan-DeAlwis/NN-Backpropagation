function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%Expanding y input into matrix
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:); % 5000 x 10 matrix

%Adding bias to Input Layer
alpha_1 = [ones(size(X,1),1) X]; % 5000 x 401 matrix

%Calculate Hidden Layer
z2 = alpha_1 * Theta1'; %5000 x 25 matrix
alpha_2 = sigmoid(z2);

%Adding bias to Hidden Layer
alpha_2 = [ones(size(alpha_2,1),1) alpha_2];

%Calculate Output Layer
z3 = alpha_2 * Theta2';
alpha_3 = sigmoid(z3); %5000 x 10 matrix

%Computing Cost Function
J = (1/m) * sum(sum((-1) * y_matrix .* log(alpha_3) - (1-y_matrix).*log(1-alpha_3)));

%Adding Regularization
J = J + ((lambda/(2*m))*...
    (sum(sum(Theta1(:,2:end).*Theta1(:,2:end))) + sum(sum(Theta2(:,2:end).*Theta2(:,2:end)))));

sGz2 = sigmoidGradient(z2);% 5000 x 25 matrix

%Backpropagation Algorithm
d3 = alpha_3 - y_matrix; %5000 x 10 matrix
d2 = d3*Theta2(:,2:end).*sGz2; %5000x10 * 10x25 .* 5000 x 25 = 5000 x 25
Delta1 = d2' * alpha_1; % 5000x25 * 5000x401
Delta2 = d3' * alpha_2; % 5000x10 * 5000x26

%Compute unregularized Gradient
Theta1_grad = Delta1/m; % 25 x 401 matrix
Theta2_grad = Delta2/m; % 10 x 26 matrix



%Implement regularization with the cost function and gradients.
Theta1(:,1) = 0;
Theta2(:,1) = 0;
Theta1 = (lambda/m)*Theta1; %25 x 401 matrix
Theta1_grad = Theta1_grad + Theta1; % 25x401 + 25x401
Theta2 = (lambda/m)*Theta2;
Theta2_grad = Theta2_grad + Theta2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
