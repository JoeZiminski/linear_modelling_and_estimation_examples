%
% Overview of ANOVA and contrast estimation
% ---------------------------------------------------------------------------------------------------------------------------------------------------

rng(0);

k = 4;           % Typically in the literature, k is used to denote number of groups, n for participants per group, N for total participants (n * k)
n = 5;
N = k * n;

%  Running an one-way ANOVA by 'hand'
% 4 groups, 1 dependent variable

% This example will show one-way ANOVA in matrix form. 
% As ANOVA is a special case of regression, this is formulated exactly as OLS in matrix form, where
%  X contains k columns in which 1 identifies the group for that row (N rows). 

% Some ANOVA background:
%
% The model is estimated as the mean of each group. The null hypothesus is that u1 == u2 == ... == uk i.e. that the variance of the means is zero.
% 
% SS - sum of squared deviation, df = degrees of freedon (e.g. n - 1), MS - mean squares (SS / df) i.e. estimate of the variance
% The total variance (MSt) is partitioned into:
%           MSb - between group, or treatment variance (variance of the means u1 .. uk) (i.e. 1/K * SUM i=1:k (xi - x_bar) where k is the number of groups)
%           MSw - within group variance i.e. average of the variance of the 4 groups, for one group (level i) 1/n * SUM j=1:n (xij - xi_bar)
%           MSe - error variance (assumed M = 0, Sigma normally distributed).
%    MSt = MSb + MSw + MSe.

% Lets setup a standard data table with k groups. We can reform this into the dependent variable y and a design matrix that codes
% the groups.

% Setup and Fit The OLS Model ------------------------------------------------------------------------------------------------------------

labels = repelem(1:k, n);  % [1, 1, ..., 2, 2, ..., 3, 3, ....]
data = array2table([labels', randn(N, 1)], 'RowNames', cellstr(string(1:N)), 'VariableNames', {'group', 'y'});  % TODO: convert this...

y =  data{:, 'y'};

% First we setup the design matrix
X = kron(eye(k),  ...
         ones(n, 1));  % the kronker product of Ik and 1n creates a N x k matrix with 1 in column
                       % of group assignment else zeros. This is clear when calculated by hand.

% Calculate the model parameters using OLS without an intercept. Note that the B are the group means and the 
% grand mean is modelled with the group means. This is called the 'cell means' approach [1].

% We can see how this extracts group means as the design matrix nulls any y not in the group, and X'X becomes the n
% e.g. X = [1 0    y = [y1   X'X = [2  0   X'y = [y1 + y2,           
%           1 0         y2          0  2]         y3 + y4]   and so [X'X]-1 * X'y becomes the means
%           0 1         y3
%           0 1]        y4]

B = inv(X' * X) * X' * y;

% Now try add the intercept. Now the design matrix is not full rank and B is not estimable. To estimate models including the intercept
% we must apply more complex design matrix, see Appendix 1.
X_ = [X, ones(N, 1)];
B_ = inv(X_' * X_) * X_' * y;

% Partition the variance and calculate the F statistic ------------------------------------------------------------------------------------

% It is useful to conceptualise the ANOVA as a model-comparison between the 'full' i.e. fully parameterised model,
% in which the variation around the means is taken, and the 'reduced' model (grand mean only). Then the F statistic can be seen as:  
% RRS = redisual sum of squares: (RSS_reduced - RSS_full) / (k - 1) 
%                                 ------------------------------------
%                                         RSS_full / (N - k)
%
% where RSS_between_groups = RSS_reduced - RSS_full

y_ = y - mean(y);

X_r = ones(N, 1); % reduced
X_f = X;          % full 

B_r = inv(X_r' * X_r) * X_r' * y_;
B_f = inv(X' * X) * X' * y_;

RSS_r = sum((y_ - X_r * B_r).^2);
RSS_f = sum((y_ - X_f * B_f).^2);

F_ = ((RSS_r - RSS_f) / DFb) / (RSS_f / DFe);

% Contrast Matrix -----------------------------------------------------------------------------------------------------------------
% https://www.fil.ion.ucl.ac.uk/~wpenny/publications/rik_anova.pdf
% https://www.marekrychlik.com/sites/default/files/05_contrasts1.pdf (great simple but detailed description of contrasts)

% The above estimates the linear model, partitions the variance and calculates the F statistic for the model. A p-value
% lookup table can be used to determine significance based on the F-statistic distribution. Post-doc comparison of group means
% can be used to determine specific difference in means. However, the heart of ANOVA is in the contrast matrix, which allows
% group-mean comparisons as well as sophisticated mutli-factor ANOVA designs. 

% Classican ANOVA ('extra sum of squares')
% Classically, different hypothesis can be tested in ANOVA by comparing the residuals a full, and reduced model
% where the hypothesis determines the two models to test. For example, if we wanted to test the hypothesis that
% B1 == 0, we can compare the full model (B1-B4) with the reduced model (B2-B4). 

B_ = inv(X' * X) * X' * y_;  % estimate b from demenated y
RSS_f = sum((y - X * B).^2);

X_ = X;
X_ = X(:, 2:end);
X_(1:5, 1:3) = 1;
B_r = inv(X_' * X_) * X_' * y_;
RSS_r = sum((y - X_ * B_r).^2);

rank_f = rank(X);
rank_r = rank(X_);

F = ((RSS_r - RSS_f) / (rank_f - rank_r)) / (RSS_f / (N - rank_f));  % note the df is the difference in model ranks

% Use of a Contrast Matrix -------------------------------------------------------------------------------------------------------------------------------------------

% A contrast matrix can be used in place of the 'extra sum of squares principle' and has the advantage
% that rather than fitting 2 models, the hypothesis are tested in a single step.

% Contrats essentially take the estimated betas and test them in the numerator with the XX variance in denominator.
% e.g. if we have 3 groups and betas, b1, b2, b3 we could test that b1 is different to the mean of b2 and b3 with
% [-1 0.5 0.5] i.e. -b1 + ((b2 + b3) / 2). This value would form the numerator while the standard error for the contrast
% would be the demoninator (e.g. like a t distribution). Alternatively, you can perform an F-test to evaluate contrasts, 
% t^2 = F. (TDO: page 16 of https://www.marekrychlik.com/sites/default/files/05_contrasts1.pdf)

% standard error of the contrast = sqrt (MSW * SUMi (ci^2 / ni)) where MSW is the MS (within groups) from the omnibus ANOVA


% First, let us use the example of a one-way ANOVA (3 levels) with one contrast testing that B0 = 0

c = [1 0 0 0]; 
B_ = inv(X' * X) * X' * y_;  % estimate b from demenated y
H = X * inv(X' * X) * X';
SSw = y' * (eye(N) - H) * y;                      % from http://users.stat.umn.edu/~helwig/notes/aov1-Notes.pdf (this is very detailed and clear)
                                                  % this is very cool, so the hat matrix is for each group block 1/k. The eye(N) - H becomes
                                                  % essentially a within-group contrast e.g. group 1, sub 1 (row 1, col 1 - 5) = 0.8 -0.2 -0.2 -0.2 -0.2. 
                                                  % We can see that this is y1 - y1/5 - y2/5 ...
                                                  % Somehow, amazingly, y'*this rsults in SSw, even though it is not the same as multiplying each yhat-y 
                                                  % it must even out somehow within the groups. Look more into this, genius!
DFw = N - k;
MSW = SSw / DFw;
c_std_error = sqrt(MSW * sum(c.^2 / n));                      % Recall the std error is the std / sqrt(n) and MSW is the variance. 
                                                              % see
                                                              % https://stats.stackexchange.com/questions/89154/general-method-for-deriving-the-standard-error
                                                              % for derivation of standard error 

t = sum(c * B) / c_std_error;

% TODO 
% Lets now do a more complex analysis, that all three groups are different. We can do this with a pair of orthogonal contrats. 
% Pairs of contrasts must be orthononal. This is calculated as the covariance (sum i=1:N ai*bi) or a'b and
% can also be interpreted as the angle between two vectors (e.g. (1, 1) and (1, -1) == 90 degrees. 
% We can also imagine each contrast as a vector in T dimensional space (where T is the number of groups, e.g.
% for 3 groups (1 -1 0) and (-1 1 0). Then the contrasts have a 90 degree angle between them. 
% Orthononality can also be interpreted as shareing no variance,
% we want to use orthononal contrasts as as complete set of orthononal contrassts perfectly particiaton SSb 
% http://www-personal.umich.edu/~gonzo/coursenotes/file3.pdf

% Run level comparison with contrasts...
% https://www.southampton.ac.uk/~cpd/anovas/datasets/Orthogonal%20contrasts.htm


% TODO -------------------------------------------------------------------------

% try ANOVA with regressor
% two - way ANOVA
% RM ANOVA
% linear mixed-effects models (fixed vs. random effects)

% MANOVA and MANOVA Contrast

% https://www.real-statistics.com/multivariate-statistics/multivariate-analysis-of-variance-manova/manova-follow-up-contrasts/
% https://online.stat.psu.edu/stat505/book/export/html/762
% https://support.minitab.com/en-us/minitab/18/help-and-how-to/modeling-statistics/anova/how-to/general-manova/interpret-the-results/key-results/
% http://www.regorz-statistik.de/en/manova_multivariate_contrasts_SPSS.html
% https://docs.tibco.com/data-science/GUID-D7CAED4A-A391-4BBD-BFD0-5DF071BA34FC.html
% https://www.lexjansen.com/wuss/2010/analy/2981_3_ANL-LIN.pdf

% Fixed vs. Random effects  -------------------------------------------------------------------------------------------------------


% Running two-way ANOVA by 'hand'
% [1] https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/GLM
% [2] Analysis of Variance, W Penny and R Henson 2006. 

