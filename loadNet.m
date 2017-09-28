function net = loadNet(img_size, use_gpu)
% Loads a Caffe 'net' object for a specific image dimensions
%
%
% Input:
% img_size: MAP Image size [Height, Width].
% use_gpu: GPU flag: use 1 if you use GPU, use 0 to run on CPU.
%
% Output:
% map: Caffe 'net' object.

%%
net_size = [3, img_size(2), img_size(1)];

caffe.reset_all();

if use_gpu
    caffe.set_mode_gpu();
    caffe.set_device(1);
else
    caffe.set_mode_cpu();
end

net_model = 'model/deploy_DAE_base.prototxt';
net_weights = 'model/DAE_sigma11.caffemodel';

FID_base = fopen(net_model, 'r');
Str_base = fread(FID_base, [1, inf]);
fclose(FID_base);
FID_net = fopen('model/deploy_DAE_resized.prototxt', 'w');
fprintf(FID_net, char(Str_base), net_size);
fclose(FID_net);
net_model = 'model/deploy_DAE_resized.prototxt';

net = caffe.Net(net_model, net_weights, 'test');
