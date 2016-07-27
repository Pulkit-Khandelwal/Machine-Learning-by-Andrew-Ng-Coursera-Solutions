function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


step= [.01; .03; .1; 0.3; 1; 3; 10; 30];
ks = size(step,1) * size(step,1);
k = 1;
storage = zeros(ks,2);
errors_store = zeros(ks,1);
for i = 1:size(step,1)
    for j = 1: size(step,1)
        model= svmTrain(X, y, step(i), @(x1, x2) gaussianKernel(x1, x2, step(j)));
        predictions = svmPredict(model,Xval);
        error = mean(double(predictions ~= yval));
        errors_store(k,1) = error;
        storage(k,:) = [step(i) step(j)];
        k = k+1;
        
    end
end
[min_error, min_index] = min(errors_store);
C = storage(min_index,1)
sigma = storage(min_index,2)

% steps = [ 0.01 0.03 0.1 0.3 1 3 10 30 ];
% minError = Inf;
% minC = Inf;
% minSigma = Inf;
% 
% for i = 1:length(steps)
%     for j = 1:length(steps)
%         currentC = steps(i);
%         currentSigma = steps(j);
%         model = svmTrain(X, y, currentC, @(x1, x2) gaussianKernel(x1, x2, currentSigma));
%         predictions = svmPredict(model, Xval);
%         error = mean(double(predictions ~= yval));
% 
%         if error < minError
%             minError = error;
%             minC = currentC;
%             minSigma = currentSigma;
%         end
%     end
% end
% 
% C = minC
% sigma = minSigma


% =========================================================================

end
