function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

n = size(X, 2);
theta1 = theta(1);
theta2_n = theta(2:n);
X1 = X(:, 1);
X2_n = X(:, 2:n);

% cost J
J = 1/m * sum(-y .* log(sigmoid(X*theta)) - (1-y) .* log(1 - sigmoid(X*theta))) + (lambda/2/m)* theta2_n' * theta2_n;

% for j = 0
grad(1) = 1/m * X1' * (sigmoid(X*theta) - y);

% for j = 1
grad(2:n) = 1/m * X2_n' * (sigmoid(X*theta) - y) + lambda/m*theta2_n;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
