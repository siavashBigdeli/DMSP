function rec = DAE_caffe(x, net)

rec = net.forward({x});
rec = rec{1};
