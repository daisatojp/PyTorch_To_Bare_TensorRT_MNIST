#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <ratio>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <cuda_runtime_api.h>
#include <NvCaffeParser.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

using namespace nvinfer1;
using namespace plugin;

namespace dnn {

namespace F = torch::nn::functional;

using at::Tensor;
using at::Device;
using at::DeviceType;
using at::IntArrayRef;
using at::IntList;
using at::ScalarType;
using at::TensorOptions;
using torch::NoGradGuard;
using torch::InferenceMode;
using torch::autograd::Variable;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::variable_list;
using torch::jit::named_module_list;
using torch::jit::named_attribute_list;
using torch::jit::named_parameter_list;
using torch::jit::IValue;
using torch::jit::script::Module;
using torch::jit::script::Method;

constexpr int64_t sidx = 0;
constexpr int64_t eidx = 9223372036854775807;

constexpr DeviceType CPU = at::DeviceType::CPU;
constexpr DeviceType CUDA = at::DeviceType::CUDA;
constexpr ScalarType Bool = at::ScalarType::Bool;
constexpr ScalarType Byte = at::ScalarType::Byte;
constexpr ScalarType Int8 = at::ScalarType::Char;
constexpr ScalarType Int = at::ScalarType::Int;
constexpr ScalarType Long = at::ScalarType::Long;
constexpr ScalarType Half = at::ScalarType::Half;
constexpr ScalarType Float = at::ScalarType::Float;

} /* dnn */

#define ASSERT(condition) \
  do { \
    if (!(condition)) { \
      std::cout << "Assertion failure: " << #condition << std::endl; \
      abort(); \
    } \
  } while (0)

struct InferDeleter {
  template <typename T>
  void operator()(T* obj) const {
    delete obj;
  }
};

template <typename T>
using TRTUniquePtr = std::unique_ptr<T, InferDeleter>;

class Logger : public ILogger {
public:
  void log(Severity severity, AsciiChar const* msg) noexcept override {
    std::cout << msg << std::endl;
  }
};

Logger gLogger;

static auto StreamDeleter = [](cudaStream_t* pStream) {
  if (pStream) {
    cudaStreamDestroy(*pStream);
    delete pStream;
  }
};

#define CHECK(status) \
  do { \
    auto ret = (status); \
    if (ret != 0) { \
      std::cout << "Cuda failure: " << ret << std::endl; \
      abort(); \
    } \
  } while (0)

std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream() {
  std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(new cudaStream_t, StreamDeleter);
  if (cudaStreamCreate(pStream.get()) != cudaSuccess) {
    pStream.reset(nullptr);
  }
  return pStream;
}

static std::shared_ptr<ICudaEngine> mEngine(nullptr);

static void constructNetwork(
  TRTUniquePtr<IBuilder>& builder,
  TRTUniquePtr<INetworkDefinition>& network,
  TRTUniquePtr<IBuilderConfig>& config);
static void build();
static void infer();

void build()
{
  TRTUniquePtr<IBuilder> builder(
    createInferBuilder(gLogger));
  ASSERT(builder);

  TRTUniquePtr<INetworkDefinition> network(
    builder->createNetworkV2(0));
  ASSERT(network);

  TRTUniquePtr<IBuilderConfig> config(
    builder->createBuilderConfig());
  ASSERT(config);

  constructNetwork(
    builder, network, config);
}

using tensor_dict = std::map<std::string, dnn::Tensor>;

tensor_dict get_tensor_dict(const dnn::Module& m) {
  auto named_parameters = m.named_parameters();
  tensor_dict dict;
  for (
    dnn::named_parameter_list::iterator itr = named_parameters.begin();
    itr != named_parameters.end(); ++itr
  ) {
    dict[(*itr).name] = (*itr).value;
    std::cout << "Found parameter (" << (*itr).name << ")" << std::endl;
  }
  return dict;
}

void constructNetwork(
  TRTUniquePtr<IBuilder>& builder,
  TRTUniquePtr<INetworkDefinition>& network,
  TRTUniquePtr<IBuilderConfig>& config)
{
  dnn::Module script = torch::jit::load("../script.pt");

  std::map<std::string, dnn::Module> modules;
  {
    auto named_modules = script.named_modules();
    for (
      dnn::named_module_list::iterator itr = named_modules.begin();
      itr != named_modules.end(); ++itr
    ) {
      modules[(*itr).name] = (*itr).value;
      std::cout << "Found module (" << (*itr).name << ")" << std::endl;
    }
  }

  ITensor* data = network->addInput(
    "input", DataType::kFLOAT, Dims3{1, 28, 28});
  ASSERT(data);

  const float scaleParam = 0.0125f;
  const Weights power{DataType::kFLOAT, nullptr, 0};
  const Weights shift{DataType::kFLOAT, nullptr, 0};
  const Weights scale{DataType::kFLOAT, &scaleParam, 1};
  IScaleLayer* scale_1 = network->addScale(
    *data, ScaleMode::kUNIFORM, shift, scale, power);
  ASSERT(scale_1);

  tensor_dict conv1_t = get_tensor_dict(modules["conv1"]);
  dnn::Tensor conv1_t_w = conv1_t["weight"].cpu().contiguous();
  dnn::Tensor conv1_t_b = conv1_t["bias"].cpu().contiguous();
  Weights conv1_w = {DataType::kFLOAT, conv1_t_w.data_ptr(), conv1_t_w.numel()};
  Weights conv1_b = {DataType::kFLOAT, conv1_t_b.data_ptr(), conv1_t_b.numel()};
  IConvolutionLayer* conv1 = network->addConvolutionNd(
    *scale_1->getOutput(0), 20, Dims{2, {5, 5}}, conv1_w, conv1_b);
  ASSERT(conv1);
  conv1->setStride(DimsHW{1, 1});

  IPoolingLayer* pool1 = network->addPoolingNd(
    *conv1->getOutput(0), PoolingType::kMAX, Dims{2, {2, 2}});
  ASSERT(pool1);
  pool1->setStride(DimsHW{2, 2});

  tensor_dict conv2_t = get_tensor_dict(modules["conv2"]);
  dnn::Tensor conv2_t_w = conv2_t["weight"].cpu().contiguous();
  dnn::Tensor conv2_t_b = conv2_t["bias"].cpu().contiguous();
  Weights conv2_w = {DataType::kFLOAT, conv2_t_w.data_ptr(), conv2_t_w.numel()};
  Weights conv2_b = {DataType::kFLOAT, conv2_t_b.data_ptr(), conv2_t_b.numel()};
  IConvolutionLayer* conv2 = network->addConvolutionNd(
    *pool1->getOutput(0), 50, Dims{2, {5, 5}}, conv2_w, conv2_b);
  ASSERT(conv2);
  conv2->setStride(DimsHW{1, 1});

  IPoolingLayer* pool2 = network->addPoolingNd(
    *conv2->getOutput(0), PoolingType::kMAX, Dims{2, {2, 2}});
  ASSERT(pool2);
  pool2->setStride(DimsHW{2, 2});

  tensor_dict fc1_t = get_tensor_dict(modules["fc1"]);
  dnn::Tensor fc1_t_w = fc1_t["weight"].cpu().contiguous();
  dnn::Tensor fc1_t_b = fc1_t["bias"].cpu().contiguous();
  Weights fc1_w = {DataType::kFLOAT, fc1_t_w.data_ptr(), fc1_t_w.numel()};
  Weights fc1_b = {DataType::kFLOAT, fc1_t_b.data_ptr(), fc1_t_b.numel()};
  IFullyConnectedLayer* ip1 = network->addFullyConnected(
    *pool2->getOutput(0), 500, fc1_w, fc1_b);
  ASSERT(ip1);

  IActivationLayer* relu1 = network->addActivation(
    *ip1->getOutput(0), ActivationType::kRELU);
  ASSERT(relu1);

  tensor_dict fc2_t = get_tensor_dict(modules["fc2"]);
  dnn::Tensor fc2_t_w = fc2_t["weight"].cpu().contiguous();
  dnn::Tensor fc2_t_b = fc2_t["bias"].cpu().contiguous();
  Weights fc2_w = {DataType::kFLOAT, fc2_t_w.data_ptr(), fc2_t_w.numel()};
  Weights fc2_b = {DataType::kFLOAT, fc2_t_b.data_ptr(), fc2_t_b.numel()};
  IFullyConnectedLayer* ip2 = network->addFullyConnected(
    *relu1->getOutput(0), 10, fc2_w, fc2_b);
  ASSERT(ip2);

  ISoftMaxLayer* prob = network->addSoftMax(*ip2->getOutput(0));
  ASSERT(prob);
  prob->getOutput(0)->setName("output");
  network->markOutput(*prob->getOutput(0));

  builder->setMaxBatchSize(1);
  config->setMaxWorkspaceSize(16 * (1 << 20)); /* 16 MB */

  auto profileStream = makeCudaStream();
  ASSERT(profileStream);
  config->setProfileStream(*profileStream);

  TRTUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
  ASSERT(plan);

  TRTUniquePtr<IRuntime> runtime{createInferRuntime(gLogger)};
  ASSERT(runtime);

  mEngine = std::shared_ptr<ICudaEngine>(
    runtime->deserializeCudaEngine(
      plan->data(), plan->size()),
    InferDeleter());
  ASSERT(mEngine);
}

void infer()
{
  TRTUniquePtr<IExecutionContext> context(mEngine->createExecutionContext());
  ASSERT(context);

  cv::Mat img = cv::imread("../1.png", cv::IMREAD_GRAYSCALE);
  dnn::Tensor input = at::from_blob(img.data, dnn::IntArrayRef({ 1, 28, 28 }), dnn::Byte);
  input = input.to(dnn::Device(dnn::CUDA, 0)).to(dnn::Float).contiguous();
  dnn::Tensor output = torch::zeros({1, 10}, dnn::Device(dnn::CUDA, 0)).to(dnn::Float).contiguous();

  std::vector<void*> buffers;
  buffers.push_back(input.data_ptr());
  buffers.push_back(output.data_ptr());

  std::vector<uint8_t> fileData(28 * 28);
  std::memcpy((void*)(fileData.data()), (void*)(img.data), 28 * 28);
  std::cout << std::endl << "Input:" << std::endl << std::endl;
  for (int i = 0; i < 28 * 28; i++)
    std::cout << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % 28) ? "" : "\n");

  bool status = context->execute(1, buffers.data());
  ASSERT(status);

  torch::save(output, "../cpp_tensor.pt");

  output = output.cpu()[0];
  std::cout << std::endl << "Output:" << std::endl << std::endl;
  float maxVal = 0.0f;
  int idx = 0;
  for (int i = 0; i < 10; i++) {
    float prob = output[i].item<float>();
    if (maxVal < prob) {
      maxVal = prob;
      idx = i;
    }
    std::cout << i << ": " << std::string(int(std::floor(prob * 10 + 0.5f)), '*') << std::endl;
  }
  std::cout << std::endl;
  ASSERT(maxVal > 0.9f);
}

int main()
{
  build();
  infer();

  return 0;
}
