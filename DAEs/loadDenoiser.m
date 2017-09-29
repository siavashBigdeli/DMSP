function DAE = loadDenoiser(denoiser, use_gpu, img_size)

switch denoiser
    case 'caffe'
        DAE = loadCaffeDenoiser(use_gpu, img_size);
    case 'matconvnet'
        DAE = loadMatDenoiser(use_gpu);
end