function [k,dk] = Laplace_BF_ift_wrapper(betalogscale,lambdalogscale,BFParams,inputbeta,hyp,x,z)
% Wrapper of ift of basis function to be compatible with gpml
% BF_ift_handle is a handle taking two inputs @(distance,BFParams)
if nargin < 5
    k = '0';
    return;
end
zempty = isempty(z);
zdiag = strcmp(z,'diag');
eqspace = range(diff(x)) < 1e-8;

% inputbeta may be on log scale
if betalogscale == 1
    beta = exp(inputbeta);
else
    beta = inputbeta;
end

if zdiag
    % Shortcut for diagonal of kernel matrix
    k = repmat(Laplace_BF_ift(0,BFParams,lambdalogscale)*beta,[length(x),1]);
else
    if zempty && eqspace
        % If equally spaced x and no z, use Toeplitz for fast computation
        tau=linspace(0,(x(2)-x(1))*(length(x)-1),length(x))';
        k = toeplitz(Laplace_BF_ift(tau,BFParams,lambdalogscale)*beta);
    else
        % If no z, but x not equally spaced, copy x to z
        if zempty
            z = x;
        end
        
        % Full explicit calculation
        m = length(x);
        n = length(z);
        tau = repmat(x,[1,n])-repmat(z',[m,1]);
        k = reshape(Laplace_BF_ift(tau(:),BFParams,lambdalogscale)*beta,m,n);
    end
end
dk = @(Q)zeros(0,1);
end
