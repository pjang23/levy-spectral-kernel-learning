function [basis_priors,J_prior] = Levy_Prior_PDF(LevyPriorStruct,BasisFunction,J,basis_params)
% Computes the prior pdf values of a set of given basis expansion parameters
beta = basis_params(1,:)';

% Prior for chi
xmin = BasisFunction.domain(1);
xmax = BasisFunction.domain(2);
chi_prior = ((xmin < basis_params(2,:)').*(xmax > basis_params(2,:)'))/(xmax-xmin);

% Prior for lambda
a_lambda = BasisFunction.hyperparams(1);
b_lambda = BasisFunction.hyperparams(2);
lambda_prior = gampdf(basis_params(3,:)', a_lambda, 1./b_lambda);

% Prior for beta and J
switch LevyPriorStruct.name
    case 'Gamma'
        gam = LevyPriorStruct.hyperparams(1);
        eta = LevyPriorStruct.hyperparams(2);
        epsilon = LevyPriorStruct.hyperparams(3);
        nu_eps_plus = gam*(xmax-xmin)*expint(epsilon);
        J_prior = poisspdf(J,nu_eps_plus);
        beta_prior = (beta).^(-1).*exp(-abs(beta)*eta)/(expint(epsilon)).*((beta*eta)>epsilon);
    case 'SymGamma'
        gam = LevyPriorStruct.hyperparams(1);
        eta = LevyPriorStruct.hyperparams(2);
        epsilon = LevyPriorStruct.hyperparams(3);
        nu_eps_plus = 2*gam*(xmax-xmin)*expint(epsilon);
        J_prior = poisspdf(J,nu_eps_plus);
        beta_prior = abs(beta).^(-1).*exp(-abs(beta)*eta)/(2*expint(epsilon)).*(abs(beta*eta)>epsilon);
    case 'Stable'
        gam = LevyPriorStruct.hyperparams(1);
        eta = LevyPriorStruct.hyperparams(2);
        epsilon = LevyPriorStruct.hyperparams(3);
        alpha = LevyPriorStruct.hyperparams(4);
        nu_eps_plus = gam*(xmax-xmin)*2/pi*gamma(alpha)*sin(pi*alpha/2)*epsilon^(-alpha);
        J_prior = poisspdf(J,nu_eps_plus);
        beta_prior = alpha*(epsilon/eta)^alpha/2*abs(beta).^(-alpha-1).*(abs(beta*eta)>epsilon);

end

basis_priors = [beta_prior'; chi_prior'; lambda_prior'];

end