function rec = DAE_mat(x, net)

rec = vl_simplenn(net, single(x),[],[],'conserveMemory',true,'mode','test');
rec = x + double(rec(end).x);
