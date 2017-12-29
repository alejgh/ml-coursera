function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
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
num_iter_c = 10;
num_iter_sigma = 10;

lowest_error = 1;

for i = 0:num_iter_c
    current_c = 0.01 * (3 * i);
    for j = 0:num_iter_sigma
        current_sigma = 0.01 * (3 * j);
        model = svmTrain(X, y, current_c, ...
            @(x1, x2) gaussianKernel(x1, x2, current_sigma));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
        if (error < lowest_error)
            C = current_c;
            sigma = current_sigma;
            lowest_error = error;
        end
    end
end

% =========================================================================

end
