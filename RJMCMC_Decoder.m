function bases = RJMCMC_Decoder(theta, BasisFunction, betalogscale, X, J)
% Computes one of the function decompositions
%
% Inputs
% samples: structure
%   Output from RJMCMC
% BasisFunction: structure
%   Contains basis function handle
% betalogscale
%   1 if MCMC done on log(beta)
%   0 if MCMC done on beta
% X: N x 1 vector
%   Evaluation points
% index: scalar
%   The index of the function to extract


for j = 1:J
    if betalogscale == 1
        bases(:,j) = BasisFunction.function(X ,theta(2:3,j)')*exp(theta(1,j)');
    else
        bases(:,j) = BasisFunction.function(X ,theta(2:3,j)')*(theta(1,j)');
    end
end

end