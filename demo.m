
addpath(genpath('DAEs'))

% select denoiser
denoiser_name = 'caffe'; % make sure matCaffe is installed and its location is added to path
% denoiser_name = 'matconvnet'; % make sure matconvnet is installed and its location is added to path

% set to 0 if you want to run on CPU (very slow)
use_gpu = 1;
% use_gpu = 0;

%% Load data

% load image and kernel
load('data/kernels.mat');

gt = double(imread('data/101085.jpg'));

kernel = kernels{1};
sigma_d = 255 * .01;

degraded = convn(gt, rot90(kernel,2), 'valid');
noise = randn(size(degraded));
degraded = degraded + noise * sigma_d;

% load denoiser for solver
params.denoiser = loadDenoiser(denoiser_name, use_gpu, size(gt));

% set parameters
params.sigma_dae = 11; % correspones to the denoiser's training standard deviation
params.num_iter = 300; % number of iterations
params.gt = gt; % to print PSNR at each iteration


%% non-blind deblurring demo

% run DMSP
restored = DMSPDeblur(degraded, kernel, sigma_d, params);

figure;
subplot(131);
imshow(gt/255); title('Ground Truth')
subplot(132);
imshow(degraded/255); title('Blurry')
subplot(133);
imshow(restored/255); title('Restored')

%% Noise-blind deblurring demo

% run DMSP noise-blind
restored_nb = DMSPDeblur(degraded, kernel, -1, params);

figure;
subplot(131);
imshow(gt/255); title('Ground Truth')
subplot(132);
imshow(degraded/255); title('Blurry')
subplot(133);
imshow(restored_nb/255); title('Restored (noise-blind)')
