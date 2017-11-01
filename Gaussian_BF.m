function [Phi] = Gaussian_BF(x, BFParams)
% Gaussian Basis Function
% Inputs:
% x = n x 1 vector of inputs
% BFParams = J x 2 matrix of basis function parameters
%   BFParams(:,1) = chi
%   BFParams(:,2) = lambda2 (square of length scale)
%
% Output:
% Phi = n x J matrix 
%   Phi(i,j) = sqrt(lambda2(j)/(2*pi))*exp(-0.5*lambda2(j)*(x(i)-chi(j))^2)  

chi = BFParams(:,1);
lambda2 = BFParams(:,2);
J = size(chi,1);
n = size(x,1);
X = repmat(x,[1,J]);
Chi = repmat(chi',[n,1]);
Lambda2 = repmat(lambda2',[n,1]);
Phi = sqrt(Lambda2/(2*pi)).*exp(-0.5*Lambda2.*(X-Chi).^2);
end
