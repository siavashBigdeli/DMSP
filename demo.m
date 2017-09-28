
% add MatCaffe path
addpath /mnt/data/siavash/caffe/matlab;

% set to 0 if you want to run on CPU (very slow)
use_gpu = 1;


%% Deblurring demo

% load image and kernel
load('kernels.mat');

gt = double(imread('101085.jpg'));

kernel = kernels{1};
sigma_d = 255 * .01;

degraded = convn(gt, rot90(kernel,2), 'valid');
noise = randn(size(degraded));
degraded = degraded + noise * sigma_d;

% load network for solver
params.num_iter=300;
params.net = loadNet(size(gt), use_gpu);
params.gt = gt;

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
