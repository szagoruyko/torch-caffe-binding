require 'caffe'

prototxt_name = '/opt/caffe/models/bvlc_reference_caffenet/deploy.prototxt'
binary_name = '/opt/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(prototxt_name, binary_name, 'test')
input = torch.randn(10,3,227,227):float()
output = net:forward(input)
gradInput = net:backward(input, output)
