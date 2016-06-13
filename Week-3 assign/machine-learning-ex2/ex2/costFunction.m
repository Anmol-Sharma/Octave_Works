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
temp=X*theta;
sig_vals=sigmoid(temp);
posi=log(sig_vals);
negi=log(1-sig_vals);

arb1=(-y).*posi;
arb2=(1-y).*negi;
J=(1/m)*sum(arb1-arb2);

for i=1:size(X,2)
  grad(i,1)=(1/m)*sum((sig_vals-y).*X(:,i));
end


% =============================================================

end
