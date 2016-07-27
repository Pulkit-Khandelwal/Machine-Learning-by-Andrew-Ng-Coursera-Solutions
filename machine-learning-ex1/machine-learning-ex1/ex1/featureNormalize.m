function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

mu = [mean(X(:,1)) mean(X(:,2))];
sigma = [std(X(:,1)) std(X(:,2))];
k=size(X);
m=k(1,1);
para1 = (X(:,1) - (mu(1,1)* ones(m,1)))./(sigma(1,1));
para2 = (X(:,2) - (mu(1,2)* ones(m,1)))./(sigma(1,2));
X_norm = [para1 para2];

% mean_X1=mean(X(:,1));
% std_X1=std(X(:,1));
% mean_X2=mean(X(:,2));
% std_X2=std(X(:,2));
% k=size(X);
% m=k(1,1); 	
% 
% 
% for i=1:m
% mu(i,1)=mean_X1;
% mu(i,2)=mean_X2;
% end
% 
% for i=1:m
% sigma(i,1)=std_X1;
% sigma(i,2)=std_X2;
% end
% 
% X_norm=(X-mu)./sigma





% ============================================================

end
