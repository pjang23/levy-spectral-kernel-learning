function [Phi] = Laplace_BF(x, BFParams, lambdalogscale)
% Laplace Basis Function
% Inputs:
% x = n x 1 vector of inputs
% BFParams = J x 2 matrix of basis function parameters
%   BFParams(:,1) = chi
%   BFParams(:,2) = lambda or log(lambda)
% lambdalogscale = 1 if KernelParams(:,2) is log(lambda2)
%                = 0 if KernelParams(:,2) is lambda2
%
% Output:
% Phi = n x J matrix 
%   Phi(i,j) = lambda(j)/2*exp(-lambda(j)*|x(i)-chi(j)|)  

chi = BFParams(:,1);
if lambdalogscale == 1
    lambda = exp(BFParams(:,2));
else
    lambda = BFParams(:,2);
end
J = size(chi,1);
n = size(x,1);
X = repmat(x,[1,J]);
Chi = repmat(chi',[n,1]);
Lambda = repmat(lambda',[n,1]);
Phi = Lambda/2.*exp(-Lambda.*abs(X-Chi));
end