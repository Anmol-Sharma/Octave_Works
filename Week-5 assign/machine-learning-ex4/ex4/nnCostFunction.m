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
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

lab=1:num_labels;
tmp_y=zeros(size(X,1),num_labels);
for i=1:size(y,1)
  tmp_y(i,:)=(y(i)==lab);
end
y=tmp_y;
X=[ones(m, 1) X];
a2=sigmoid(X*Theta1');
a2=[ones(m,1) a2];
a3=sigmoid(a2*Theta2');

posi=log(a3);
negi=log(1-a3);
arb1=sum(((-y).*posi),2);
arb2=sum(((1-y).*negi),2);

J=(1/m)*sum(arb1-arb2);

teta1=Theta1(:,2:end);
teta2=Theta2(:,2:end);

var1=sum(sum(teta1.^2,2));
var2=sum(sum(teta2.^2,2));
regul=(lambda/(2*m))*(var1+var2);
J=J+regul;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients

del_1=zeros(hidden_layer_size,size(X,2));
del_2=zeros(num_labels,hidden_layer_size+1);
D1=zeros(size(del_1));
D2=zeros(size(del_2));
for t=1:m
  a1=X(t,:);
  z2=a1*Theta1';
  a2=sigmoid(z2);
  a2=[1, a2];
  a3=sigmoid(a2*Theta2');
  delta3=a3-y(t,:);
  delta2=(delta3*Theta2).*sigmoidGradient([1,z2]);
  delta2=delta2(2:end);
  del_1=del_1+(delta2'*a1);
  del_2=del_2+(delta3'*a2);
end
D1=(1./m)*del_1 + (lambda/m)*[zeros(size(Theta1,1), 1) Theta1(:, 2:end)];
D2=(1./m)*del_2 + (lambda/m)*[zeros(size(Theta2,1), 1) Theta2(:, 2:end)];

Theta1_grad=D1;
Theta2_grad=D2;

grad = [Theta1_grad(:) ; Theta2_grad(:)];
