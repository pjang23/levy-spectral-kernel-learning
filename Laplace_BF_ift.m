function [k] = Laplace_BF_ift(tau, BFParams,lambdalogscale)
% Inverse Fourier transform of Laplace basis function
% Inputs:
% x = n x 1 vector of inputs
% BFParams = J x 2 matrix of basis function parameters
%   BFParams(:,1) = chi
%   BFParams(:,2) = lambda
% lambdalogscale = 1 if KernelParams(:,2) is log(lambda)
%                = 0 if KernelParams(:,2) is lambda
%
% Output:
% k = n x J matrix 
%   k = ift( s(i,j) = lambda(j)/2*exp(-lambda(j)*|x(i)-chi(j)|))  

chi = BFParams(:,1);
if nargin < 3
    lambda = BFParams(:,2);
elseif lambdalogscale == 1
    lambda = exp(BFParams(:,2));
else
    lambda = BFParams(:,2);
end
J = size(chi,1);
n = size(tau,1);
Tau = repmat(tau,[1,J]);
Chi = repmat(chi',[n,1]);
Lambda = repmat(lambda',[n,1]);
k = Lambda.^2./(Lambda.^2 + 4*pi^2*Tau.^2).*cos(2*pi*Chi.*Tau);
end
