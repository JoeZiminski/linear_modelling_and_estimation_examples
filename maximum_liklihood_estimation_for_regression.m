% Walkthough of maximium liklihood estimation for linear regression
% (followed here) https://machinelearningmastery.com/linear-regression-with-maximum-likelihood-estimation/
% https://medium.com/swlh/linear-regression-and-maximum-likelihood-1dcb9435c71e - this is a nice walkthrough of equivilence with sum of squares, as a test should be able to do this from memory

% Bivariate
% The problem: We are looking for the parameters b0 and b1 in the equation y = b0 + b1 * X + error that scale the straight line such that it is 
% the best possible fit to the data. 

data = mvnrnd([0, 0], [1, 0.5; 0.5, 1], 1000);
X = data(:, 1);
y = data(:, 2);

% Regression is tractable by linear algebra. The derivative = 0 of the loss function can be found with a close-form solution
X_int = [ones(size(X)), X];  % add a column of ones as the intercept for OLS;
B = inv(X_int' * X_int) * X_int' * y;

scatter(X, y); hold on
plot(X, B(1) + B(2) * X);

% But less us imagine it is not tractable. Then, we have to generate a loss function and try all combinations of b0 and b1 to see how this loss function changes. 
% Here we define the problem as the difference between the data and models generated with different parameters.
% In the popular least squares optimisation, this is defined as the sum of squares, sum((yhat - y)^2). 
% The results are very similar to the closed form solution.
% Note the second link above demonstrates how the maximum liklihood approach is equivilent to minimising the sum of squares.

b0_range = linspace(-1, 1, 1000);
b1_range = linspace(-1, 1, 1000);
sum_of_squared_loss = NaN(length(b0_range), length(b1_range));

for i = 1:length(b0_range)
    for j = 1:length(b1_range)
                
        y_hat = b0_range(i) + b1_range(j) * X;
        ss = sum((y - y_hat).^2);
        sum_of_squared_loss(i, j) = ss;
        
    end
end

figure;
plot3(b0_range, b1_range, sum_of_squared_loss);
[row, col] = ind2sub([1000, 1000], ...
                     find(sum_of_squared_loss == min(min(sum_of_squared_loss))));  % find the coord of the minimum
                 
b0 = b0_range(row);   
b1 = b1_range(col);
scatter(X, y); hold on
plot(X, b0 + b1 * X);

fprintf('The coefficients from the closed form solution are: b0 = %.5f, b1 = %.5f\nWhile from the gradient method b0 = %.5f, b1 = %.5f\n', B(1), B(2), b0, b1);
                 
% We can also conceptualise the problem as a Maximum Liklihood problem. Here we say we wish to maximise the conditional probability of observing
% the data given a specific probability distribution and its parameters. 
% P(X | theta)

% Note that it is conditional on a probability distribution, NOT the form of our model.
% To calculate it we can find the product of the marginal probabilities, x1 ... xn. 
% For computation, we can also take the log of this probability quick is easier and reduces numerical instability. We can also minimize the negative
% log liklihood rather than maximise the log liklihood

% To calculate for linear regression, we first conceptualise the problem as
% log P(X | h) where h is our choice of model parameters. See
% https://stats.stackexchange.com/questions/310067/can-you-use-bayes-theorem-to-transform-a-likelihood-function-into-a-probability for detail 

% We make the assumption that the observations are iid and the noise is Gaussian with zero mean without heterosketdasciticity.
% The residuals being independent means we can calculate the probability of p(h | x) = p(h | x1) * p(h | x2) .... * p(h | xn)
% Now what is the probably of xi | h?
% Recall that the residuals of the model are assumed to be drawn from a gaussian distribution with m = 0, 
% and estimate the variance with 1/n * (y - (b0 + b1x)^2

% so the probability of h given xi is the gaussian distribution formula with sigma = v (a constant),
% mu = 0 and x = (y - (b0 + b1x)) (i.e. the residual that we are calculating the probability for.
% e.g. https://www.thoughtco.com/normal-distribution-bell-curve-formula-3126278

% f(x) = ( 1 / sqrt(2 * pi * sigma^2)) * exp((-(y - mu)^2) / (2 * sigma^2));

% We can use this as a liklihood function where mu is the prediction from the model with a given set of coefficients 
% and sigma is a fixed constant

% f(x) = (1 / sqrt(2 * pi * sigma^2)) * exp(- (y(i) - (b0 + b1 * X(i)))^2 / (2 * sigma^2));

% First try taking without taking the log. This works (at least in this case) but can cause numerical issues and is very slow:

sigma = 0.5;
liklihood_range = NaN(length(b0_range), length(b1_range));

tic;
for i = 1:length(b0_range)
    for j = 1:length(b1_range)
        
        liklihood = 1;
        for z = 1:length(y)
           marginal_liklihood = (1 / sqrt(2 * pi * sigma^2)) * exp(- (y(z) - (b0_range(i) + b1_range(j) * X(z)))^2 * (1/2 * sigma^2));
            
           liklihood = liklihood * marginal_liklihood;

        end
        
        liklihood_range(i, j) = liklihood;
        
    end
end
fprintf('time taken with liklihood: %0.4f seconds \n', toc);
[row, col] = ind2sub([1000, 1000], ...
                     find(liklihood_range == max(max(liklihood_range))));  % find the coord of the minimum
b0 = b0_range(row);   
b1 = b1_range(col);
fprintf('The coefficients from the closed form solution are: b0 = %.5f, b1 = %.5f\nWhile from the liklihood method b0 = %.5f, b1 = %.5f\n', B(1), B(2), b0, b1);

% Next, calculate the log of the lilklihood function (quick revision of logarithms shows how to do this) -----------------------------------
% (note everything except terms involving xi can come out of the sum)
% We see this is well matched to the OLS version (and matches the numerical sos version exactly)

sigma = 0.5;
log_liklihood_range = NaN(length(b0_range), length(b1_range));

tic;
for i = 1:length(b0_range)
    for j = 1:length(b1_range)
               
        liklihood = 0;
        for z = 1:length(y)

           marginal_liklihood = (log(1 / (sqrt(2 * pi * sigma^2)))) - ( (y(z) - (b0_range(i) + b1_range(j) * X(z)))^2 * (1/2 * sigma^2) );

           liklihood = liklihood + marginal_liklihood;

        end
        
        log_liklihood_range(i, j) = liklihood;
        
    end
end
fprintf('time taken with log liklihood: %0.4f seconds \n', toc);

[row, col] = ind2sub([1000, 1000], ...
                     find(log_liklihood_range == max(max(log_liklihood_range))));  % find the coord of the maximum
b0 = b0_range(row);   
b1 = b1_range(col);
fprintf('The coefficients from the closed form solution are: b0 = %.5f, b1 = %.5f\nWhile from the log liklihood method b0 = %.5f, b1 = %.5f\n', B(1), B(2), b0, b1);

