local ffi = require 'ffi'
local C = caffe.C
local Net = torch.class('caffe.Net', 'nn.Module')

local function typecheck(i)
  if(i:type() ~= 'torch.FloatTensor') then print 'Only FloatTensor supported' end
end

function Net:init(prototxt_name, binary_name)
  self.handle = ffi.new('void*[1]')
  local old_handle = self.handle
  C['init'](self.handle, prototxt_name, binary_name)
  if(self.handle[1] == old_handle[1]) then
    print 'Unsuccessful init'
  end
  self.output = torch.Tensor()
  self.gradInput = torch.Tensor()
  self:float()
end

function Net:forward(input)
  typecheck(input)
  C['do_forward'](self.handle, input:cdata(), self.output:cdata())
  return self.output
end

function Net:updateGradInput(input, gradOutput)
  typecheck(input)
  typecheck(gradOutput)
  c['do_backward'](self.handle, input:cdata(), gradOutput:cdata(), self.gradInput:cdata())
  return self.gradInput
end

function Net:reset()
  C['reset'](self.handle)
end

function Net:setModeCPU()
  C['set_mode_cpu']()
end

function Net:setModeGPU()
  C['set_mode_gpu']()
end

function Net:setPhaseTrain()
  C['set_phase_train']()
end

function Net:setPhaseTest()
  C['set_phase_test']()
end

function Net:setDevice(device_id)
  C['set_device'](device_id)
end

