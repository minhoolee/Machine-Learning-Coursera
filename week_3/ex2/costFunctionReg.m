function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Hypothesis function
h = sigmoid(X * theta);

% Calculate inner expresions when y = 1 or y = 0
positive = y .* log(h);
negative = (1 - y) .* log(1 - h);

% Correct cost function for overfitting, do not regularize theta (1)
thetaT = theta;
thetaT(1) = 0;
correction = (lambda/(2 * m)) * sum (thetaT .^ 2);

% Calculate cost function
J = ((-1/m) * sum (positive + negative)) + correction;

% Calculate gradient (partial derivative). 
% Theta (1) is not regularized
grad = ((1/m) * X' * (h - y)) + ((lambda / m) * thetaT);

% =============================================================

end
