function [fmugf,fs2gf,ymugf,ys2gf] = computePostMeanVar(Xstar,nPredict,samples,useSKI)
% For each of the last nPredict posterior kernel samples from RJ-MCMC, 
% computes the conditional mean and variance of the GP at test inputs
% Xstar. If nPredict = 1, then outputs the results for the MAP kernel.

% ystar = fstar + eps,  eps ~ N(0,sigma^2)
% fstar|X,y ~ N(fmugf,fs2gf)
% ystar|X,y ~ N(ymugf,ys2gf)


numSamples = length(samples.log_Posterior);
if nPredict > numSamples
    nPredict = numSamples;
    fprintf('Warning: Only %d RJ-MCMC samples available.',numSamples);
end
if useSKI == 0
    fprintf('Calculating predictive mean and variance with exact covariances\n')
else
    fprintf('Calculating predictive mean and variance with SKI approximation\n')
end

% Compute GP Predictions over the last nPredict Random Kernels
fmugf = zeros(length(Xstar),1);
fs2gf = zeros(length(Xstar),1);
ymugf = zeros(length(Xstar),1);
ys2gf = zeros(length(Xstar),1);
for s = numSamples-nPredict+1:numSamples
    fprintf('Predictive Sample %d\n',s-numSamples+nPredict)
    if nPredict == 1
        % If only one element for Bayes Average, take the MAP estimate
        [~,s] = max(real(samples.log_Posterior));
    end
    J_s = samples.J(s);
    JJ = [0; cumsum(samples.J*3)];
    theta = reshape(samples.theta(JJ(s)+1:JJ(s+1)), 3, J_s);
    if betalogscale == 1
        beta = exp(theta(1,:))';
    else
        beta = theta(1,:)';
    end
    BFParams = theta(2:end,:)';
    
    tic
    % % Recover optimal kernel
    if useSKI == 1
        % Use SKI to approximate kernel (for large training data sets)
        gpcov = {SKIParams.bf_ift_wrapper,betalogscale,lambdalogscale,BFParams,theta(1,:)'};
        gpmean = {@meanZero};
        lik = {@likGauss};
        hyp.cov = [];
        hyp.mean = [];
        hyp.lik = 0.5*log(sigma2);
        covg = {@apxGrid,{gpcov},{SKIParams.xg}};% grid prediction
    %     xg2 = linspace(min(X),max(Xstar),10000)';
    %     covg = {@apxGrid,{gpcov},{xg2}};% grid prediction
    
        opt.cg_maxit = 20000;
        opt.cg_tol = 1e-8;
        opt.cg_showit = 1;
        opt.use_pred_var = 1; % Flag for Perturb-and-MAP (1 to activate, 0 to deactivate)
        if opt.use_pred_var == 1
            opt.pred_var = 100;
        elseif isfield(opt,'pred_var')
            opt = rmfield(opt,'pred_var');
        end
        [postg,~] = infGrid(hyp,gpmean,covg,lik,X,y,opt);  % fast grid prediction
        [fmugf_s,fs2gf_s,ymugf_s,ys2gf_s] = postg.predict(Xstar);
    else
        % Compute exact covariance (for small training data sets)
        tauX=linspace(0,(X(2)-X(1))*(length(X)-1),length(X))';
        tauXstar=linspace(0,(Xstar(2)-Xstar(1))*(length(Xstar)-1),length(Xstar))';
        if betalogscale == 1
            beta = exp(theta(1,:)');
        else
            beta = theta(1,:)';
        end
        % Recover optimal kernel
        [k_analytical_xx] = BasisFunction.function_ift(tauX, BFParams)*beta;

        % Apply kernel to test points
        [k_analytical_zz] = BasisFunction.function_ift(tauXstar, BFParams)*beta;
        K_Synth_xx = toeplitz(k_analytical_xx);
        K_Synth_zz = toeplitz(k_analytical_zz);
        K_Synth_zx = zeros(length(Xstar), length(X));

        for i = 1:length(X)
            for j = 1:length(Xstar)
                tau = X(i) - Xstar(j);
                K_Synth_zx(j,i) = BasisFunction.function_ift(tau, BFParams)*beta;
            end
        end
        fmugf_s = K_Synth_zx * ((K_Synth_xx + sigma2*eye(ntrain)) \ (y));
        fs2gf_s = diag(K_Synth_zz - K_Synth_zx * ((K_Synth_xx + sigma2*eye(ntrain)) \ K_Synth_zx'));
        ymugf_s = fmugf_s;
        ys2gf_s = fs2gf_s + sigma2;
    end
    toc
    
    % Posterior predictive
    % ystar = fstar + eps,  eps ~ N(0,sigma^2)
    % fstar|X,y ~ N(fmugf,fs2gf)
    % ystar|X,y ~ N(ymugf,ys2gf)
    fmugf = fmugf + fmugf_s;
    fs2gf = fs2gf + fs2gf_s;
    ymugf = ymugf + ymugf_s;
    ys2gf = ys2gf + ys2gf_s;
end
fmugf = 1/nPredict*fmugf;
fs2gf = 1/nPredict*fs2gf;
ymugf = 1/nPredict*ymugf;
ys2gf = 1/nPredict*ys2gf;

end