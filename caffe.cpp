#include <string>
#include <vector>

#include <TH/TH.h>
#include "caffe/caffe.hpp"

extern "C" 
{
void init(void* handle[1], const char* param_file, const char* model_file, const char* phase);
void do_forward(void* handle[1], THFloatTensor* bottom, THFloatTensor* output);
void do_backward(void* handle[1], THFloatTensor* gradOutput, THFloatTensor* gradInput);
void reset(void* handle[1]);
void set_mode_cpu();
void set_mode_gpu();
void set_phase_train();
void set_phase_test();
void set_device(int device_id);
}

using namespace caffe;  // NOLINT(build/namespaces)

void init(void* handle[1], const char* param_file, const char* model_file, const char* phase_name)
{
  Phase phase;
  if (strcmp(phase_name, "train") == 0) {
    phase = TRAIN;
  } else if (strcmp(phase_name, "test") == 0) {
    phase = TEST;
  } else {
    THError("Unknown phase.");
  }
  Net<float>* net_ = new Net<float>(string(param_file), phase);
  if(model_file != NULL)
    net_->CopyTrainedLayersFrom(string(model_file));
  handle[1] = net_;
}


void do_forward(void* handle[1], THFloatTensor* bottom, THFloatTensor* output) {
  Net<float>* net_ = (Net<float>*)handle[1];
  const vector<Blob<float>*>& input_blobs = net_->input_blobs();
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    CHECK_EQ(bottom->size[0]*bottom->size[1]*bottom->size[2]*bottom->size[3], input_blobs[i]->count())
        << "MatCaffe input size does not match the input size of the network";
    const float* data_ptr = THFloatTensor_data(bottom);
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(input_blobs[i]->count(), data_ptr,
          input_blobs[i]->mutable_cpu_data());
      break;
    case Caffe::GPU:
      caffe_copy(input_blobs[i]->count(), data_ptr,
          input_blobs[i]->mutable_gpu_data());
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }
  const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled();
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    // internally data is stored as (width, height, channels, num)
    // where width is the fastest dimension
    THFloatTensor_resize4d(output, output_blobs[i]->num(), output_blobs[i]->channels(), output_blobs[i]->height(), output_blobs[i]->width());
    float* data_ptr = THFloatTensor_data(output);
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(output_blobs[i]->count(), output_blobs[i]->cpu_data(),
          data_ptr);
      break;
    case Caffe::GPU:
      caffe_copy(output_blobs[i]->count(), output_blobs[i]->gpu_data(),
          data_ptr);
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }
}

void do_backward(void* handle[1], THFloatTensor* gradOutput, THFloatTensor* gradInput)
{
  Net<float>* net_ = (Net<float>*)handle[1];
  const vector<Blob<float>*>& output_blobs = net_->output_blobs();
  const vector<Blob<float>*>& input_blobs = net_->input_blobs();
  // First, copy the output diff
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    const float* const data_ptr = THFloatTensor_data(gradOutput);
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(output_blobs[i]->count(), data_ptr,
          output_blobs[i]->mutable_cpu_diff());
      break;
    case Caffe::GPU:
      caffe_copy(output_blobs[i]->count(), data_ptr,
          output_blobs[i]->mutable_gpu_diff());
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }
  // LOG(INFO) << "Start";
  net_->Backward();
  // LOG(INFO) << "End";
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    // internally data is stored as (width, height, channels, num)
    // where width is the fastest dimension
    THFloatTensor_resize4d(gradInput, input_blobs[i]->num(), input_blobs[i]->channels(), input_blobs[i]->height(), input_blobs[i]->width());
    float* data_ptr = THFloatTensor_data(gradInput);
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(input_blobs[i]->count(), input_blobs[i]->cpu_diff(), data_ptr);
      break;
    case Caffe::GPU:
      caffe_copy(input_blobs[i]->count(), input_blobs[i]->gpu_diff(), data_ptr);
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }
}



void read_mean(const char* mean_file_path, THFloatTensor* mean_tensor)
{
    std::string mean_file(mean_file_path);
    Blob<float> data_mean;
    LOG(INFO) << "Loading mean file from" << mean_file;
    BlobProto blob_proto;
    bool result = ReadProtoFromBinaryFile(mean_file.c_str(), &blob_proto);

    data_mean.FromProto(blob_proto);
    THFloatTensor_resize4d(mean_tensor, data_mean.num(), data_mean.channels(), data_mean.height(), data_mean.width());
    float* data_ptr = THFloatTensor_data(mean_tensor);
    caffe_copy(data_mean.count(), data_mean.cpu_data(), data_ptr);
}

void reset(void* handle[1])
{
  Net<float>* net_ = (Net<float>*)handle[1];
  if (net_) {
    delete net_;
    LOG(INFO) << "Network reset, call init before use it again";
  }
}

void set_mode_cpu() {
  Caffe::set_mode(Caffe::CPU);
}

void set_mode_gpu() {
  Caffe::set_mode(Caffe::GPU);
}

void set_device(int device_id) {
  Caffe::SetDevice(device_id);
}
