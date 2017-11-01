% Load up gpml
addpath gpml
startup

%% Compute Prior Covariance Samples based on initial hyperparameters
npriorkernels = 10;
fullyBayes = 1;
if fullyBayes==1
    gam = gamrnd(a_gam_0,1/b_gam_0,npriorkernels,1);
else
    gam = gam_0*ones(npriorkernels,1);
end
nu_eps = gam*(domain(2)-domain(1))*expint(eps);
sampJ = poissrnd(nu_eps,npriorkernels,1);

nbasisparams = sum(sampJ);
if fullyBayes==1
    eta = 1./gamrnd(a_eta_0,1/b_eta_0,nbasisparams,1);
else
    eta = eta_0*ones(nbasisparams,1);
end
b = exprnd(1/eta(1),1000,1);
betaprop = b + eps/eta(1);
u = rand(1000,1);
g = @(b,eta,eps)1./b.*exp(-b*eta)/expint(eps);
const = exp(-eps)/(eps*expint(eps));
sampbeta = [];
i = 0;
curBF = 1;
while length(sampbeta) < nbasisparams
    i = i+1;
    if i == 1000
        b = exprnd(1/eta(curBF),1000,1);
        betaprop = b + eps/eta(curBF);
        u = rand(1000,1);
        i = 1;
    end
    if u(i) <= g(betaprop(i),eta(curBF),eps)/(const*eta(curBF)*exp(-eta(curBF)*b(i)))
        sampbeta = [sampbeta; betaprop(i)];
        if length(sampbeta) < nbasisparams
            curBF = curBF+1;
            b = exprnd(1/eta(curBF),1000,1);
            i = 1;
        end
    end
end
samplambda = gamrnd(a_lam_0,1/b_lam_0,nbasisparams,1);
sampchi = 0.2*rand(nbasisparams,1);
priorParam = [sampbeta'; sampchi'; samplambda'];

Xstar = [xtrain+0.5*(xtrain(2)-xtrain(1)); xtest];
tauXstarPrior=linspace(0,(Xstar(2)-Xstar(1))*(length(Xstar)-1),5*length(Xstar))';

priorKernelValues = zeros(length(tauXstarPrior),npriorkernels);
for s = 1:npriorkernels
    tic;
    J_s = sampJ(s);
    JJ = [0; cumsum(sampJ*3)];
    theta = reshape(priorParam(JJ(s)+1:JJ(s+1)), 3, J_s);
    beta = exp(theta(1,:)');
    BFParams = theta(2:end,:)';
    
    % Apply kernel to test points
    priorKernelValues(:,s) = BasisFunction.function_ift(tauXstarPrior, BFParams)*beta;
end
empcov = zeros(ntrain,1);
empcovtau = (0:(ntrain-1))'*(Xstar(2)-Xstar(1));
for j = 1:ntrain
    empcov(j) = 1/ntrain*sum((y(1:end-j+1)).*(y(j:end)));
end

%% Compute Posterior Covariance Samples from airline fit data
postKernelValues = zeros(2*length(Xstar),numSamples);
tauXstar=linspace(0,(Xstar(2)-Xstar(1))*(length(Xstar)-1),2*length(Xstar))';
for s = 1:numSamples
    tic;
    J_s = samples.J(s);
    JJ = [0; cumsum(samples.J*3)];
    theta = reshape(samples.theta(JJ(s)+1:JJ(s+1)), 3, J_s);
    beta = exp(theta(1,:)');
    BFParams = theta(2:end,:)';
    
    % Apply kernel to test points
    postKernelValues(:,s) = BasisFunction.function_ift(tauXstar, BFParams)*beta;
end

figure(3); clf
subplot(1,2,1)
box on
grid on
hold on
title('Prior Covariance Samples')
plot(empcovtau,empcov,'k','Linewidth',3)
plot(tauXstarPrior,priorKernelValues)
hold off
xlim([0,50])
xlabel('\tau (Months)')
ylabel('K(\tau)')
ylim([-4000,6000])
subplot(1,2,2)
box on
grid on
hold on
plot(empcovtau,empcov,'k','Linewidth',3)
plot(tauXstar,postKernelValues(:,1:24:250))
title('Posterior Covariance Samples')
xlim([0,50])
xlabel('\tau (Months)')
ylabel('K(\tau)')
hold off
set(gcf, 'position', [50 500 2000 500])
set(gcf, 'PaperPositionMode','auto')

% print(gcf, '-dpng', '-r300', 'Airline Covariance.png')

% priorSemivariogram = repmat(priorKernelValues(1,1:npriorkernels),[length(Xstar),1]) - priorKernelValues(:,1:npriorkernels);
% empSemivariogram = empcov(1) - empcov;
% postSemivariogramFull = repmat(postKernelValues(1,:),[length(Xstar),1])-postKernelValues;
% postSemivariogram = postSemivariogramFull(:,1:24:250);
% 
% figure(4); clf
% subplot(1,2,1)
% box on
% grid on
% hold on
% plot(empcovtau,empSemivariogram,'k','Linewidth',3)
% plot(tauXstar,priorSemivariogram)
% xlim([0,50])
% ylim([0,20000])
% xlabel('\tau (Months)')
% ylabel('K(0)-K(\tau)')
% hold off
% subplot(1,2,2)
% box on
% grid on
% hold on
% plot(empcovtau,empSemivariogram,'k','Linewidth',3)
% plot(tauXstar,postSemivariogram)
% xlim([0,50])
% xlabel('\tau (Months)')
% ylabel('K(0)-K(\tau)')
% hold off
% set(gcf, 'position', [50 200 2000 500])
% set(gcf, 'PaperPositionMode','auto')
% % print(gcf, '-dpng', '-r300', 'Airline Semivariogram.png')
