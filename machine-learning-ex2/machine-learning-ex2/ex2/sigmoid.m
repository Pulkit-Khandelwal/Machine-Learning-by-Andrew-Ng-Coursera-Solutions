function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

s= ones(size(z)) + (exp(-1 * (z)));
for i = 1:size(g,1)
    for j = 1:size(g,2)
    g(i,j) =  1/s(i,j);
    end
end

% =============================================================

end
