caffe = {}
local ffi = require 'ffi'

ffi.cdef[[
void init(void* handle[1], const char* param_file, const char* model_file, const char* phase_name);
void do_forward(void* handle[1], THFloatTensor* bottom, THFloatTensor* output);
void do_backward(void* handle[1], THFloatTensor* gradOutput, THFloatTensor* gradInput);
void reset(void* handle[1]);
void set_mode_cpu();
void set_mode_gpu();
void set_device(int device_id);
]]

caffe.C = ffi.load(package.searchpath('libtcaffe', package.cpath))
