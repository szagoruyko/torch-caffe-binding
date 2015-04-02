torch-caffe-binding
===================

A short binding to use Caffe as a module in Torch7. Has the same functionality as MATLAB bindings.

You have to have installed and built Caffe, then do this:

```bash
CAFFE_DIR=/*path-to-caffe-root*/ luarocks install caffe
```

Forward and backward are supported:

```lua
require 'caffe'

net = caffe.Net('deploy.prototxt', 'bvlc_alexnet.caffemodel', 'test')
input = torch.FloatTensor(10,3,227,227)
output = net:forward(input)

gradOutput = torch.FloatTensor(10,1000,1,1)
gradInput = net:backward(input, gradOutput)
```

Use can also use it inside a network as nn.Module, for example:

```lua
require 'caffe'

model = nn.Sequential()
model:add(caffe.Net('deploy.prototxt', 'bvlc_alexnet.caffemodel', 'test'))
model:add(nn.Linear(1000,1))
```

To load Caffe networks in Torch7 without having Caffe installed use this:
https://github.com/szagoruyko/loadcaffe
