% EM Algorithm Implementation
% https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm

% Let x = (x1, x2, ..., xn) be a sample of n independent observations from
% a mixture of two multivariate normal distributions of dimension d, and
% let z = (z1, z2, ..., zn) be the latent variables that determine the
% component from which the observation origionates.

% Xi | (Z = 1) ~ Nd(u1, s1) and Xi | (Zi = 2) ~ Nd(u2, s2)

% P(Zi = 1) = tau and P(Zi = 2) = 1 - tau

% Note that this says Zi is a 1 x N indicator variable indicating from
% which gaussian distribution each datapoint Xi is drawn from. The
% probability the datapoint is drawn from gaussian 1 is tau and gaussian
% 2 is 1 - tau.

d = 2;     
n = 1000;  % ensure event
num_gaussians = 2;

mean_1 = 1;
sigma_1 = 0.25;
mean_2 = 5;
sigma_2 = 0.75;


x = [mvnrnd([mean_1, mean_1], [sigma_1, 0.15; 0.15, sigma_1], n/2);  ...  eye(d) * sigma_1
     mvnrnd([mean_2, mean_2],  [sigma_2, -0.15; -0.15, sigma_2], n/2)];
    
scatter(x(:, 1), x(:, 2));
title('Mixed Multivariate Gaussians');

% Problem Setup ===========================================================
%
% The aim is to estimate the unknown parametrs representing the mixing
% value between the Gaussians and the means and covariances of each:

% theta = (tau, u1, u2, s1, s2)

% with the 'incomplete-data liklihood function (?) is:

% L(theta, x) = times_i=1:n sum_j=1:2 tau_j f(x; uj, sj)

% where f is the probability density function of the multivariate normal.

% i.e. the liklihood of the parameters (mixing, and means / variances of
% the gaussians) given the data = multiply for each datapoint, the sum of 
% the gaussian PDF with mean uj, sigma sj over x multiplied by the
% probability that Xi is drawn from gaussian j. So for each datapoint,
% we calculate the probability it is drawn from Gaussian 1 and Gaussian 2
% weighted to the liklihood of the distribution it comes from. So this is 
% fitting all parameters at once.

% and the 'complete-data' liklihood function is:

% L(theta; x, z) = p(x, z | theta) = times_i=1:n times(j=1:2 [f(xi; uj, sj) tauj] I(zi = j)

% where I is an indicator random variable (not 100% sure exactly what this
% is indicator variable is saying).

% but overall this says that the liklihood of observing the parameters
% given the data and our indicator variable for what datapoint belongs to
% what distribution= the probability of observing the data AND each
% individual datapoint belonging to one of the distributions, GIVEN theta
% (by bayes).

% EM Running ==============================================================

% E-Step. Estimate the missing variables in the dataset.
% M-Step. Maximize the parameters of the model in the presence of the data.

% E-Step. Estimate the expected value for each latent variable.
% M-Step. Optimize the parameters of the distribution using maximum likelihood.

% Where Z is our latency variable of indicator 1, 2 for Gaussian 1, 2

% (Expectation) Assign each data point to a cluster probabilistically. 
%               In this case, we compute the probability it came from the 
%               red cluster and the yellow cluster respectively.

% (Maximization) Update the parameters for each cluster (weighted mean 
%                location and variance-covariance matrix) based on the 
%                points in the cluster (weighted by their probability assigned in the first step).

% Okay so this makes a lot of sense. We estiamte the mean and distribution
% of the two clusters based on the samples assigned to each cluster. Then
% we update the assignment based on the new parameter values.
    
% Now first we need to estimate the latent variable from the starting
% estimates. The latent variable is not [0, 1] but defined in terms of
% probability. In this case it takes the form of a N x J matrix of 
% disjoint probabilities that N belongs to j = 1 or j = 2.

% Tj,i at step t = P(Z = j | Xi = x;theta(t))
%
% i.e. The probability that Zi belongs to j given the probability of x
% given the parameters (for each distribution).

% Starting Estimates - Interesting it will not converge if these are the
% same e.g. the means
tau = [0.1, 0.9];
u = [0.1, 0.2];
s = {eye(d) * 0.1, eye(d) * 0.1};
Q_t_m_1 = 0;


% 1) ask about how to handle the non-scalar u estimates (one for x, one for y)
% 2) ask how to make it converge more in close form solution

for iter = 1:1000

    T = NaN(n, num_gaussians);

    for i = 1:n

        for j = 1:num_gaussians

            T(i, j) = ( tau(j) * mvnpdf(x(i, :), u(j), s{j}) ) /  (  tau(1) * mvnpdf(x(i, :), u(1), s{1}) +  tau(2) * mvnpdf(x(i, :), u(2), s{2}) ) ;

        end

    end

    % Now we have an estimation of our latent variable, we can calculate the
    % optimisation function over all parameters. See the wikipedia for the 
    % derivation of Q.

    % Q(theta | theta(t))is objective function across all parameters
    % given the parameters at previous time (t), which generated T. So
    % theoretically we would interate over every combination of the parameters
    % (i.e. tau, u1, u2, s1, s2) and calculate Q at least given Z. Then we
    % would take the maximum and find the parameters at this point.

    % However, because this would take a long time and a lot of space (e.g. for
    % trying 100 values of each we need to calculate 100.^5 Q) they can be
    % solved analytically.

    % Because the terms in Q calculation are linear we can find them indivudally!
    Q = 0;
    for i = 1:n
        for j = 1:num_gaussians

            Q = Q + T(i, j) * log(tau(j)) - 0.5 * log( det(s{j}) ) - 0.5 * (x(i, :) - u(j)) * pinv(s{j}) * (x(i, :) - u(j))' - 0.5 * d * log(2 * pi);

        end
    end
    
    if abs(Q - Q_t_m_1) < 0.000001
        disp("COMPLETE");
        return
    end
    
    tau = sum(T(:, 1)) / n;
    u1 = (sum(T(:, 1) .* x))/sum(T(:, 1)); u1 = mean(u1);  % TODO: check this 
    u2 = (sum(T(:, 2) .* x))/sum(T(:, 2)); u2 = mean(u2);

    % s1
    sum_ = 0;
    for i = 1:n
        sum_ = sum_ + T(i, 1) * (x(i, :) - u1)' * (x(i, :) - u1);
    end
    s1 = sum_ / sum(T(:, 1));

    % s2
    sum_ = 0;
    for i = 1:n
        sum_ = sum_ + T(i, 2) * (x(i, :) - u2)' * (x(i, :) - u2);
    end
    s2 = sum_ / sum(T(:, 2));


    u = [u1; u2];
    s = {s1, s2};
    tau = [tau, 1 - tau];

    Q_t_m_1 = Q;
        
end












% ---------------------------------------------------------------------------------------
% Notes 
% ---------------------------------------------------------------------------------------

% very basic implementation (I think might be wrong) just to show that what
% we are actually doingwhen finding theta | theta (t) is like iterating
% over the entire parameter space to find the combination of params
% that maximimses the liklihood of the params given params at previous step
% and the latent variable we calculated on this step. 

div = 100;

u1 = 0:10/div:10;
u2 = 0:10/div:10;
%try_mean = [u1; u2]';

s1 = {};
s2 = {};
for gen_sigma = 10/div:10/div:10+10/div
    
    s1{end+1} = eye(2) * gen_sigma;
    s2{end+1} = eye(2) * gen_sigma;

end
%try_sigma = [s1; s2]';

try_tau_1 = 0:1/div:1;
try_tau_2 = 1 - try_tau_1;
%try_tau = [try_tau_1; try_tau_2]'

Q = NaN(div, div, div, div, div);
   
for i_u1 = 1:div
    disp([num2str(i_u1) " out of " num2str(div)]);
    
    for i_u2 = 1:div
        
        for i_s1 = 1:div
            
            for i_s2 = 1:div
                
                for i_tau = 1:div
                    
                    
                    % calc Q
                    it_Q = 0;
                    for i_n = 1:n
                        
                        j_1 = T(i, 1) * log(try_tau_1(i_tau)) - 0.5*log( det(s1{i_s1}) ) - 0.5*(x(i, :) - u1(i_u1)) * inv(s1{i_s1}) * (x(i, :) - u1(i_u1))' - 0.5 * d * log(2 * pi);  % note transpose
                        j_2 = T(i, 1) * log(try_tau_2(i_tau)) - 0.5*log( det(s2{i_s2}) ) - 0.5*(x(i, :) - u2(i_u2)) * inv(s2{i_s2}) * (x(i, :) - u2(i_u2))' - 0.5 * d * log(2 * pi);
                        
                        it_Q = it_Q + (j_1 + j_2);  % obviously this is not extendable to increased d, just a first try
                    end
                
                    Q(i_u1, i_u2, i_s1, i_s2, i_tau) = it_Q;
                    
                end
                
            end
            
        end
        
    end
    
end

[~, I] = max(Q, [], 'all', 'linear') % lol completely wrong
[i1, i2, i3, i4, i5] = ind2sub(size(Q), I);
u1(i1)
u2(i2)
s1{i3}
s2{i4}
try_tau_1(i5)

%
% which equals bayes theorem where:

% P(Zi = j) = P(tau)    (makes sense)
% P(Xi = Xi; theta | Zj = j) = normal distribution on x, with parametrs uj, sigma j at time t

% P(Xi = Xi ; theta(t)) = the sum of the two gaussian distributions on x with parameters theta(t)
%weighted by the probability is drawn from both distributions (disjoint events, makes
% sense)

% so basically this says the probability that the probability datapoint Xi
% belongs to gaussian j is this probability given the probabiltiy of Xi
% given the parameters, which is bayes law for the probability Xi is from
% gaussian J. NOTE: just realised that bayes is just a ratio of the probability
% of A AND B, (PA | B) P(B)  to total probability of B.












