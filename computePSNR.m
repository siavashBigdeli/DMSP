function psnr = computePSNR(im1, im2, pad)
% Computes peak signal-to-noise ratio between two images
%
%
% Input:
% im1: First image in range of [0, 255].
% im1: Second image in range of [0, 255].
% pad: Scalar radius to exclude boundaries from contributing to PSNR computation.
%
%
% Output: PSNR

im1_u = uint8(im1(pad+1:end-pad, pad+1:end-pad,:));
im2_u = uint8(im2(pad+1:end-pad, pad+1:end-pad,:));
imdff = double(im1_u) - double(im2_u);
rmse = sqrt(mean(imdff(:).^2));
psnr = 20*log10(255/rmse);

