function DAE = loadMatDenoiser(use_gpu)


load('DAEs/matconvnet_std11/DAE_sigma11.mat')
net_obj.layers = net;
if use_gpu
    net_obj = vl_simplenn_move(net_obj, 'gpu');
else
    net_obj = vl_simplenn_tidy(net_obj);
end

DAE = @(x) DAE_mat(x, net_obj);
