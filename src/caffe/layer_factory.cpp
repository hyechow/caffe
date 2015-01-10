#include <string>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/vision_layers.hpp"
#if defined(_MSC_VER) || defined(__MACH__)
#include "caffe/common_layers.hpp"
#endif

namespace caffe {

// Get convolution layer according to engine.
template <typename Dtype>
Layer<Dtype>* GetConvolutionLayer(
    const LayerParameter& param) {
  ConvolutionParameter_Engine engine = param.convolution_param().engine();
  if (engine == ConvolutionParameter_Engine_DEFAULT) {
    engine = ConvolutionParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = ConvolutionParameter_Engine_CUDNN;
#endif
  }
  if (engine == ConvolutionParameter_Engine_CAFFE) {
    return new ConvolutionLayer<Dtype>(param);
#ifdef USE_CUDNN
  } else if (engine == ConvolutionParameter_Engine_CUDNN) {
    return new CuDNNConvolutionLayer<Dtype>(param);
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(CONVOLUTION, GetConvolutionLayer);

// Get pooling layer according to engine.
template <typename Dtype>
Layer<Dtype>* GetPoolingLayer(const LayerParameter& param) {
  PoolingParameter_Engine engine = param.pooling_param().engine();
  if (engine == PoolingParameter_Engine_DEFAULT) {
    engine = PoolingParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = PoolingParameter_Engine_CUDNN;
#endif
  }
  if (engine == PoolingParameter_Engine_CAFFE) {
    return new PoolingLayer<Dtype>(param);
#ifdef USE_CUDNN
  } else if (engine == PoolingParameter_Engine_CUDNN) {
    PoolingParameter p_param = param.pooling_param();
    if (p_param.pad() || p_param.pad_h() || p_param.pad_w() ||
        param.top_size() > 1) {
      LOG(INFO) << "CUDNN does not support padding or multiple tops. "
                << "Using Caffe's own pooling layer.";
      return new PoolingLayer<Dtype>(param);
    }
    return new CuDNNPoolingLayer<Dtype>(param);
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(POOLING, GetPoolingLayer);

// Get relu layer according to engine.
template <typename Dtype>
Layer<Dtype>* GetReLULayer(const LayerParameter& param) {
  ReLUParameter_Engine engine = param.relu_param().engine();
  if (engine == ReLUParameter_Engine_DEFAULT) {
    engine = ReLUParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = ReLUParameter_Engine_CUDNN;
#endif
  }
  if (engine == ReLUParameter_Engine_CAFFE) {
    return new ReLULayer<Dtype>(param);
#ifdef USE_CUDNN
  } else if (engine == ReLUParameter_Engine_CUDNN) {
    return new CuDNNReLULayer<Dtype>(param);
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(RELU, GetReLULayer);

// Get sigmoid layer according to engine.
template <typename Dtype>
Layer<Dtype>* GetSigmoidLayer(const LayerParameter& param) {
  SigmoidParameter_Engine engine = param.sigmoid_param().engine();
  if (engine == SigmoidParameter_Engine_DEFAULT) {
    engine = SigmoidParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = SigmoidParameter_Engine_CUDNN;
#endif
  }
  if (engine == SigmoidParameter_Engine_CAFFE) {
    return new SigmoidLayer<Dtype>(param);
#ifdef USE_CUDNN
  } else if (engine == SigmoidParameter_Engine_CUDNN) {
    return new CuDNNSigmoidLayer<Dtype>(param);
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(SIGMOID, GetSigmoidLayer);

// Get softmax layer according to engine.
template <typename Dtype>
Layer<Dtype>* GetSoftmaxLayer(const LayerParameter& param) {
  SoftmaxParameter_Engine engine = param.softmax_param().engine();
  if (engine == SoftmaxParameter_Engine_DEFAULT) {
    engine = SoftmaxParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = SoftmaxParameter_Engine_CUDNN;
#endif
  }
  if (engine == SoftmaxParameter_Engine_CAFFE) {
    return new SoftmaxLayer<Dtype>(param);
#ifdef USE_CUDNN
  } else if (engine == SoftmaxParameter_Engine_CUDNN) {
    return new CuDNNSoftmaxLayer<Dtype>(param);
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(SOFTMAX, GetSoftmaxLayer);

// Get tanh layer according to engine.
template <typename Dtype>
Layer<Dtype>* GetTanHLayer(const LayerParameter& param) {
  TanHParameter_Engine engine = param.tanh_param().engine();
  if (engine == TanHParameter_Engine_DEFAULT) {
    engine = TanHParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = TanHParameter_Engine_CUDNN;
#endif
  }
  if (engine == TanHParameter_Engine_CAFFE) {
    return new TanHLayer<Dtype>(param);
#ifdef USE_CUDNN
  } else if (engine == TanHParameter_Engine_CUDNN) {
    return new CuDNNTanHLayer<Dtype>(param);
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(TANH, GetTanHLayer);

#if defined(_MSC_VER) || defined(__MACH__)

REGISTER_LAYER_CLASS_MSC_VER(ABSVAL, AbsValLayer);
REGISTER_LAYER_CLASS_MSC_VER(ACCURACY, AccuracyLayer);
REGISTER_LAYER_CLASS_MSC_VER(ARGMAX, ArgMaxLayer);
REGISTER_LAYER_CLASS_MSC_VER(BNLL, BNLLLayer);
REGISTER_LAYER_CLASS_MSC_VER(CONCAT, ConcatLayer);
REGISTER_LAYER_CLASS_MSC_VER(CONTRASTIVE_LOSS, ContrastiveLossLayer);
REGISTER_LAYER_CLASS_MSC_VER(DATA, DataLayer);
REGISTER_LAYER_CLASS_MSC_VER(DROPOUT, DropoutLayer);
REGISTER_LAYER_CLASS_MSC_VER(DUMMY_DATA, DummyDataLayer);
REGISTER_LAYER_CLASS_MSC_VER(ELTWISE, EltwiseLayer);
REGISTER_LAYER_CLASS_MSC_VER(EUCLIDEAN_LOSS, EuclideanLossLayer);
REGISTER_LAYER_CLASS_MSC_VER(EXP, ExpLayer);
REGISTER_LAYER_CLASS_MSC_VER(FLATTEN, FlattenLayer);
REGISTER_LAYER_CLASS_MSC_VER(HINGE_LOSS, HingeLossLayer);
REGISTER_LAYER_CLASS_MSC_VER(IM2COL, Im2colLayer);
REGISTER_LAYER_CLASS_MSC_VER(IMAGE_DATA, ImageDataLayer);
REGISTER_LAYER_CLASS_MSC_VER(INFOGAIN_LOSS, InfogainLossLayer);
REGISTER_LAYER_CLASS_MSC_VER(INNER_PRODUCT, InnerProductLayer);
REGISTER_LAYER_CLASS_MSC_VER(LRN, LRNLayer);
REGISTER_LAYER_CLASS_MSC_VER(MEMORY_DATA, MemoryDataLayer);
REGISTER_LAYER_CLASS_MSC_VER(MULTINOMIAL_LOGISTIC_LOSS, MultinomialLogisticLossLayer);
REGISTER_LAYER_CLASS_MSC_VER(MVN, MVNLayer);
REGISTER_LAYER_CLASS_MSC_VER(POWER, PowerLayer);
REGISTER_LAYER_CLASS_MSC_VER(SIGMOID_CROSS_ENTROPY_LOSS, SigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS_MSC_VER(SILENCE, SilenceLayer);
REGISTER_LAYER_CLASS_MSC_VER(SLICE, SliceLayer);
REGISTER_LAYER_CLASS_MSC_VER(SOFTMAX_LOSS, SoftmaxWithLossLayer);
REGISTER_LAYER_CLASS_MSC_VER(SPLIT, SplitLayer);
REGISTER_LAYER_CLASS_MSC_VER(THRESHOLD, ThresholdLayer);
REGISTER_LAYER_CLASS_MSC_VER(WINDOW_DATA, WindowDataLayer);

bool InitLayerFactory()
{
	// As mentioned in header this is only to force the Visual Studio compiler to 
	// instantiate the static variables so the layers get registered. (Brute force)
	REGISTER_LAYER_CREATOR_MSC_VER(CONVOLUTION, GetConvolutionLayer);
	REGISTER_LAYER_CREATOR_MSC_VER(POOLING, GetPoolingLayer);
	REGISTER_LAYER_CREATOR_MSC_VER(RELU, GetReLULayer);
	REGISTER_LAYER_CREATOR_MSC_VER(SIGMOID, GetSigmoidLayer);
	REGISTER_LAYER_CREATOR_MSC_VER(SOFTMAX, GetSoftmaxLayer);
	REGISTER_LAYER_CREATOR_MSC_VER(TANH, GetTanHLayer);

	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(ABSVAL, AbsValLayer);
	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(ACCURACY, AccuracyLayer);
	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(ARGMAX, ArgMaxLayer);
	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(BNLL, BNLLLayer);
	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(CONCAT, ConcatLayer);
	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(CONTRASTIVE_LOSS, ContrastiveLossLayer);
	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(DATA, DataLayer);
	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(DROPOUT, DropoutLayer);
	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(DUMMY_DATA, DummyDataLayer);
	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(ELTWISE, EltwiseLayer);
	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(EUCLIDEAN_LOSS, EuclideanLossLayer);
	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(EXP, ExpLayer);
	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(FLATTEN, FlattenLayer);
	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(HINGE_LOSS, HingeLossLayer);
	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(IM2COL, Im2colLayer);
	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(IMAGE_DATA, ImageDataLayer);
	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(INFOGAIN_LOSS, InfogainLossLayer);
	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(INNER_PRODUCT, InnerProductLayer);
	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(LRN, LRNLayer);
	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(MEMORY_DATA, MemoryDataLayer);
	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(MULTINOMIAL_LOGISTIC_LOSS, MultinomialLogisticLossLayer);
	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(MVN, MVNLayer);
	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(POWER, PowerLayer);
	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(SIGMOID_CROSS_ENTROPY_LOSS, SigmoidCrossEntropyLossLayer);
	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(SILENCE, SilenceLayer);
	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(SLICE, SliceLayer);
	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(SOFTMAX_LOSS, SoftmaxWithLossLayer);
	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(SPLIT, SplitLayer);
	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(THRESHOLD, ThresholdLayer);
	REGISTER_LAYER_CREATOR_CLASS_MSC_VER(WINDOW_DATA, WindowDataLayer);	

	return true;
}

#endif
// Layers that use their constructor as their default creator should be
// registered in their corresponding cpp files. Do not registere them here.
}  // namespace caffe
