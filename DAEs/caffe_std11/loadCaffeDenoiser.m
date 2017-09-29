function DAE = loadCaffeDenoiser(use_gpu, img_size)


net_size = [3, img_size(2), img_size(1)];

caffe.reset_all();

if use_gpu
    caffe.set_mode_gpu();
    caffe.set_device(0);
else
    caffe.set_mode_cpu();
end

net_model = 'DAEs/caffe_std11/deploy_DAE_base.prototxt';
net_weights = 'DAEs/caffe_std11/DAE_sigma11.caffemodel';

FID_base = fopen(net_model, 'r');
Str_base = fread(FID_base, [1, inf]);
fclose(FID_base);
FID_net = fopen('DAEs/caffe_std11/deploy_DAE_resized.prototxt', 'w');
fprintf(FID_net, char(Str_base), net_size);
fclose(FID_net);
net_model = 'DAEs/caffe_std11/deploy_DAE_resized.prototxt';

net = caffe.Net(net_model, net_weights, 'test');

DAE = @(x) DAE_caffe(x, net);
