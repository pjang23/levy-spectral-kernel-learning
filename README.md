# Lévy Spectral Kernel Learning

MATLAB code for spectral kernel learning using Lévy process priors.

The main file is test_script.m, which loads data and calls all other functions. 

The script is divided into individual sections which guide the user through initialization procedures, calls RJ-MCMC, computes predictive distributions for unobserved test points, and plots diagnostics.

Below are instructions to use the code. As you go through the script, press ctrl+enter to run the code of the subsection.

%% 1. Load up gpml

This section always needs to be run once at the start, as it gives access to the Gaussian Processes for Machine Learning (GPML) toolbox.



%% 2. Load up data

This section is where the data is to be loaded. 'airline.mat' contains the training and testing data for the airline passenger experiment, with the training inputs and outputs stored in 'xtrain' and 'ytrain', and the withheld testing inputs and outputs stored in 'xtest' and 'ytest'.

If the user wishes to supply their own input data, then they should use the same variable names.



%% 3. De-mean training data

As the MCMC algorithm is for a zero-mean Gaussian process, the actual y-values supplied to the MCMC must be de-meaned. The mean can then be added back in during the prediction phase. 

The most straightforward approach is to subtract the sample mean, but one could also subtract more complicated mean functions such as linear or quadratic trends. The 'mean_trend' variable stores a handle of the design matrix used to build the mean function.
% mean_trend = @(x)[ones(size(x,1)] to subtract sample mean
% mean_trend = @(x)[ones(size(x,1),x] to subtract linear trend
% mean_trend = @(x)[ones(size(x,1),x,x.^2] to subtract quadratic trend



%% 4. Hyperparameter initialization

Though the user is free to supply their own initialization, it may be difficult to understand what are reasonable guesses for parameters and hyperparameters.

The script provides an automated initialization procedure to construct a reasonable guess based on the empirical spectral density of the training data. First a Gaussian mixture is fit to the empirical spectrum by the EM algorithm. Next the Gaussian components are converted into Laplace components by least squares. Lastly, priors and hyperpriors are tuned based on the parameters of the initial mixture.


%% 4a. Set initial number of basis functions

Before a Gaussian mixture can be fit, the user should supply an initial number of mixture components. A good guess can be made by inspecting the number of peaks in the empirical spectrum.


%% 4b. Find good initialization by fitting Gaussian mixture on empirical spectrum

Calls 'initSMhypersadvanced.m', a function by Andrew Gordon Wilson which fits a Gaussian mixture to the empirical spectrum of the data. After the Gaussian mixture is fit, the Gaussian components are converted to Laplace components by least squares.


%% 4c. Initial Spectrum Plot

A visualization which allows the user to diagnose the initial fitted spectrum. From here, the user is free to tweak the basis function parameters to get a more suitable result.

However, note that this spectrum only reflects the training data (not the withheld testing data), so a perfect fit of the empirical spectrum will not necessarily yield the best generalization performance.


%% 4d. Flags for whether to run MCMC on the log of parameters

These flag variables allow the user to perform RJ-MCMC on the log of parameter values. This can be useful if the user expects parameter values to vary over different orders of magnitude, as the log-scale more easily allows simultaneous values of different orders of magnitude.


%% 4e. Tune Levy process hyperparameters based on initial spectrum parameters

Finds reasonable settings of hyperpriors by maximum likelihood on the initial values of beta, lambda, and chi. From here, one can tweak the hyperpriors if different ranges of J, beta, lambda, and chi are desired. For example, one could multiply b_lam_0 by a constant smaller than 1 to make the prior mean of lambdas larger. This will yield thinner basis functions in the spectrum, and therefore longer range extrapolations of frequencies.

One also sets the basis function here. If a different basis function from the Laplace is desired, then the user will need to supply a function and its corresponding inverse Fourier transform.


% 4f. Summary to aid in Hyperparameter Tuning

The prior means of J, beta, and lambda are provided to help the user diagnose their hyperprior tuning.



%% 5. Setup and call RJ-MCMC

Here the user sets up the parameters for the reversible jump MCMC algorithm and supplies the inputs.


%% 5a. RJ-MCMC parameter tuning

'sigma2' is the noise variance inherent to 'y'. Larger values of sigma2 mean more of the variation in y is treated as noise, while smaller values of sigma2 mean more of the variation in y is treated as part of an underlying function of interest.

The RJ-MCMC algorithm uses Metropolis-Hastings to accept/reject proposals, and the proposal step sizes are set relative to the size of the initial parameters. If there are too many/too few rejections, the user can adjust step sizes accordingly, and a diagnostic is provided in section 5d. to check the rates of acceptance/rejection.

The RJ-MCMC algorithm has three move types (Birth, Death, and Update) where a basis function is added, removed, or updated respectively. The user can adjust the relative proportions of these move types.

The RJ-MCMC algorithm is also fully-Bayesian can allocate a proportion of steps to hyperparameter updates.


%% 5b. Parameters for Structured Kernel Interpolation (SKI):

For datasets which are too large for exact Gaussian process calculations (n ~ 10^4), one could resort to structured kernel interpolation. The setup is described in detail in the code.

Set 'useSKI' to 1 to active SKI, and 0 to deactivate SKI.


%% 5c. Call RJ-MCMC to sample Levy Kernel Process posterior

Calls the RJ-MCMC function and outputs a posterior sample of kernel parameters.


%% 5d. Compute acceptance probabilities

Used to diagnose the mixing of the MCMC run. If acceptance rates are too high, then raise the proposal step size, and if acceptance rates are too low, then lower the proposal step size.


