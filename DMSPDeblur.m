function res = DMSPDeblur(degraded, kernel, sigma_d, params)
% Implements stochastic gradient descent (SGD) Bayes risk minimization for image deblurring described in:
% "Deep Mean-Shift Priors for Image Restoration" (http://home.inf.unibe.ch/~bigdeli/DMSPrior.html)
% S. A. Bigdeli, M. Jin, P. Favaro, M. Zwicker, Advances in Neural Information Processing Systems (NIPS), 2017
%
% Input:
% degraded: Observed degraded RGB input image in range of [0, 255].
% kernel: Blur kernel (internally flipped for convolution).
% sigma_d: Noise standard deviation. (set to -1 for noise-blind deblurring)
% params: Set of parameters.
% params.net: The DAE Network object loaded from MatCaffe.
%
% Optional parameters:
% params.sigma_net: The standard deviation of the network training noise. default: 11
% params.num_iter: Specifies number of iterations.
% params.mu: The momentum for SGD optimization. default: 0.9
% params.alpha the step length in SGD optimization. default: 0.1
%
%
% Outputs:
% res: Solution.


if ~any(strcmp('net',fieldnames(params)))
    error('Need a DAE network in params.net!');
end

if ~any(strcmp('sigma_net',fieldnames(params)))
    params.sigma_net = 11;
end

if ~any(strcmp('num_iter',fieldnames(params)))
    params.num_iter = 300;
end

if ~any(strcmp('mu',fieldnames(params)))
    params.mu = .9;
end

if ~any(strcmp('alpha',fieldnames(params)))
    params.alpha = .1;
end


disp(params)

pad = floor(size(kernel)/2);
res = padarray(degraded, pad, 'replicate', 'both');

step = zeros(size(res));

if any(strcmp('gt',fieldnames(params)))
    psnr = computePSNR(params.gt, res, pad);
    disp(['Initialized with PSNR: ' num2str(psnr)]);
end

for iter = 1:params.num_iter
    if any(strcmp('gt',fieldnames(params)))
        disp(['Running iteration: ' num2str(iter)]);
        tic();
    end
    
    % compute prior gradient
    input = res(:,:,[3,2,1]); % Switch channels for caffe    
    noise = randn(size(input)) * params.sigma_net;
    rec = params.net.forward({input+noise});

    prior_grad = input - rec{1};
    prior_grad = prior_grad(:,:,[3,2,1]);
    
    % compute data gradient
    map_conv = convn(res,rot90(kernel,2),'valid');
    data_err = map_conv-degraded;
    data_grad = convn(data_err,kernel,'full');

    
    if sigma_d<0
        sigma2 = 2*params.sigma_net*params.sigma_net;
        lambda = (numel(degraded))/(sum(data_err(:).^2) + numel(degraded)*sigma2*sum(kernel(:).^2));
        relative_weight = (lambda)/(lambda + 1/params.sigma_net/params.sigma_net);
    else
        relative_weight = (1/sigma_d/sigma_d)/(1/sigma_d/sigma_d + 1/params.sigma_net/params.sigma_net);
    end
    
    % sum the gradients
    grad_joint = data_grad*relative_weight + prior_grad*(1-relative_weight);
   
    % update
    step = params.mu * step - params.alpha * grad_joint;
    res = res + step;
    res = min(255,max(0,res));

    if any(strcmp('gt',fieldnames(params)))
        psnr = computePSNR(params.gt, res, pad);
        disp(['PSNR is: ' num2str(psnr) ', iteration finished in ' num2str(toc()) ' seconds']);
    end
    
end
