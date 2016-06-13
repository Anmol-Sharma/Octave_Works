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


temp=X*theta;
sig_vals=sigmoid(temp);
posi=log(sig_vals);
negi=log(1-sig_vals);
arb1=(-y).*posi;
arb2=(1-y).*negi;

teta=theta(2:length(theta));
regul=(lambda/(2*m))*sum(teta.^2);

J=(1/m)*sum(arb1-arb2)+regul;

grad(1,1)=((1/m)*sum((sig_vals-y).*X(:,1)));
for i=2:size(X,2)
  grad(i,1)=(1/m)*sum((sig_vals-y).*X(:,i))+(lambda/m)*theta(i);
end


% =============================================================

end
