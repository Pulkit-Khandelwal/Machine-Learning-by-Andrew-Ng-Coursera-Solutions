function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

s= ones(m,1)+ (exp(-1 *(X*theta)));
g= zeros(m,1);
for i = 1:m
    g(i,1) =  1/s(i,1);
end
k = 1/m;
J = k * sum((( -1 * y) .* (log(g))) - ((ones(m,1)-y) .* (log (ones(m,1)- g))));
diff = g - y;
grad = k*(X' * diff);
% =============================================================

end
