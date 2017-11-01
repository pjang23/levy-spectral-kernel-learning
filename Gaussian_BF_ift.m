function [k] = Gaussian_BF_ift(x, BFParams)
% Inverse Fourier transform of Gaussian basis function
% Inputs:
% x = n x 1 vector of inputs
% BFParams = J x 2 matrix of basis function parameters
% BFParams(:,1) = chi
% BFParams(:,2) = lambda
%
% Output:
% Phi = n x J matrix 
%   Phi(i,j) = exp(-0.5*lambda2(j)*(x(i)-chi(j))^2)  

chi = BFParams(:,1);
lambda2 = BFParams(:,2);
J = size(chi,1);
n = size(x,1);
Tau = repmat(x,[1,J]);
Chi = repmat(chi',[n,1]);
Lambda2 = repmat(lambda2',[n,1]);
k =  sqrt(2*pi./abs(Lambda2)).*exp(-2*pi^2*Tau.^2./Lambda2).*cos(2*pi*Chi.*Tau);
end
