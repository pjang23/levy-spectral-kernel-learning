function [basis_params] = Levy_Basis_Sampler(LevyPrior, BasisFunction)
% Outputs a sample vector of basis function parameters
%
% Inputs:
% LevyPrior - structure
%   LevyPrior.name - string: Gamma, SymGamma, or Stable
%   LevyPrior.hyperparams - vector: [gamma, eta, epsilon] or [gamma, eta, epsilon, alpha]
% BasisFunction - structure
%   BasisFunction.domain - 2 x 1: [xmin, xmax]
%   BasisFunction.function - function handle
%   BasisFunction.hyperparams - vector: [a_lambda, b_lambda]
%
% Outputs:
% basis_params: - vector of parameters
%   basis_params(1) - coefficient
%   basis_params(2:end) - kernel parameters

hyp = LevyPrior.hyperparams;
xmin = BasisFunction.domain(1); xmax = BasisFunction.domain(2);
a_lambda = BasisFunction.hyperparams(1); b_lambda = BasisFunction.hyperparams(2);
params.LevyPriorName = LevyPrior.name;
params.LevyPriorHyperparams = LevyPrior.hyperparams;

switch LevyPrior.name
    case 'Gamma'
        gam = hyp(1); eta = hyp(2); epsilon = hyp(3);
        u = unifrnd(gamcdf(epsilon/eta, 1, 1./eta), 1);
        beta = gaminv(u, 1, 1./eta);
        
    case 'SymGamma'
        gam = hyp(1); eta = hyp(2); epsilon = hyp(3);
        u = unifrnd(gamcdf(epsilon/eta, 1, 1./eta), 1);
        beta = sign(unifrnd(-1,1)) .* gaminv(u, 1, 1./eta);
    case 'Stable'
        gam = hyp(1); eta = hyp(2); epsilon = hyp(3); alpha = hyp(4); 
        nu = gam * (xmax - xmin) * 2/pi * gamma(alpha) * sin(pi*alpha/2)*epsilon^(-alpha);
        u = unifrnd(stblcdf(epsilon/eta,alpha,0,gam/eta^alpha,0), 1);
        beta = sign(unifrnd(-1,1)) .* stblinv(u,alpha,0,gam/eta^alpha,0);
end
        
% Sample kernel parameters
chi = unifrnd(xmin, xmax);
lambda = gamrnd(a_lambda, 1./b_lambda);
basis_params = [beta; chi; lambda];