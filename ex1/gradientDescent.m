function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    costNow = computeCost(X, y, theta);
    fprintf('Iteration: %d; \tCost: %.4f\n', iter, costNow)
    J_history(iter) = costNow;
    numFeature = length(theta);
    tmp = zeros(numFeature, 1);
    for j = 1:numFeature
        tmp(j) = theta(j) - 1/(2*m)* alpha*sum((X*theta - y).*X(:,j));
    end

    theta = tmp;
%     disp('theta:')
%     disp(theta)
end
end

