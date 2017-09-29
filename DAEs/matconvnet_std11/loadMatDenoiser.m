function DAE = loadMatDenoiser(use_gpu)


load('DAEs/matconvnet/DAE_sigma11.mat')
net.layers = net;
if use_gpu
    net = vl_simplenn_move(net, 'gpu');
else
    net = vl_simplenn_tidy(net);
end

DAE = @(x) DAE_mat(x, net);