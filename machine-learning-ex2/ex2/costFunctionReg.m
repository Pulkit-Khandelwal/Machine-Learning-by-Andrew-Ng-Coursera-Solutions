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

s= ones(m,1)+ (exp(-1 *(X*theta)));
g= zeros(m,1);
for i = 1:m
    g(i,1) =  1/s(i,1);
end
k = 1/m;
J = (k * sum((( -1 * y) .* (log(g))) - ((ones(m,1)-y) .* (log (ones(m,1)- g))))) + ((lambda/(2*m))* (sum(theta(2:(size(theta)),1) .* theta(2:(size(theta)),1))));
diff = g - y;
grad = (k*(X' * diff))+ ((lambda/m)* theta);
grad(1) = k* sum(X(:,1) .* diff(:,1));


% =============================================================

end
