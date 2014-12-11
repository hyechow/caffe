/**
 * @brief A layer factory that allows one to register layers.
 * During runtime, registered layers could be called by passing a LayerParameter
 * protobuffer to the CreateLayer function:
 *
 *     LayerRegistry<Dtype>::CreateLayer(param);
 *
 * There are two ways to register a layer. Assuming that we have a layer like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeLayer : public Layer<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is defined in the protobuffer as
 *
 *   enum LayerType {
 *     // other definitions
 *     AWESOME = 46,
 *   }
 *
 * If the layer is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *
 *    REGISTER_LAYER_CLASS(AWESOME, MyAwesomeLayer);
 *
 * Or, if the layer is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Layer<Dtype*> GetMyAwesomeLayer(const LayerParameter& param) {
 *      // your implementation
 *    }
 *
 * (for example, when your layer has multiple backends, see GetConvolutionLayer
 * for a use case), then you can register the creator function instead, like
 *
 * REGISTER_LAYER_CREATOR(AWESOME, GetMyAwesomeLayer)
 *
 * Note that each layer type should only be registered once.
 */

#ifndef CAFFE_LAYER_FACTORY_H_
#define CAFFE_LAYER_FACTORY_H_

#include <map>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Layer;

template <typename Dtype>
class LayerRegistry {
 public:
  typedef Layer<Dtype>* (*Creator)(const LayerParameter&);
  typedef std::map<LayerParameter_LayerType, Creator> CreatorRegistry;

  static CreatorRegistry& Registry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

  // Adds a creator.
  static void AddCreator(const LayerParameter_LayerType& type,
                         Creator creator) {
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 0)
        << "Layer type " << type << " already registered.";
    registry[type] = creator;
  }

  // Get a layer using a LayerParameter.
  static Layer<Dtype>* CreateLayer(const LayerParameter& param) {
    LOG(INFO) << "Creating layer " << param.name();
    const LayerParameter_LayerType& type = param.type();
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 1);
    return registry[type](param);
  }

 private:
  // Layer registry should never be instantiated - everything is done with its
  // static variables.
  LayerRegistry() {}
};


template <typename Dtype>
class LayerRegisterer {
 public:
  LayerRegisterer(const LayerParameter_LayerType& type,
                  Layer<Dtype>* (*creator)(const LayerParameter&)) {
    // LOG(INFO) << "Registering layer type: " << type;
    LayerRegistry<Dtype>::AddCreator(type, creator);
  }
};

#ifndef _MSC_VER

#define REGISTER_LAYER_CREATOR(type, creator)                                  \
  static LayerRegisterer<float> g_creator_f_##type(                            \
      LayerParameter_LayerType_##type, creator<float>);                        \
  static LayerRegisterer<double> g_creator_d_##type(                           \
      LayerParameter_LayerType_##type, creator<double>)			

#define REGISTER_LAYER_CLASS(type, clsname)                                    \
  template <typename Dtype>                                                    \
  Layer<Dtype>* Creator_##clsname(const LayerParameter& param) {               \
    return new clsname<Dtype>(param);                                          \
  }                                                                            \
  REGISTER_LAYER_CREATOR(type, Creator_##clsname)

#else
	// VS compiler optimises the static variables out of the library. Not sure why but
	// guess it thinks the variables aren't used internally and have no external interface
	// so it doesn't need to bother with them. i.e. it ignroes the static variables
	// created by REGISTER_LAYER_CREATOR macro. Couldn't get round it with the linker
	// settings so had to brute force it with a LayerRegisterInit function, ugly I know.
	// This needs to be called at the begining of the exe or dll using the library

	#define REGISTER_LAYER_CREATOR(type, creator)             ;               

	#define REGISTER_LAYER_CREATOR_MSC_VER(type, creator)					\
	  static LayerRegisterer<float> g_creator_f_##type(                     \
        LayerParameter_LayerType_##type, creator<float>);                   \
	  static LayerRegisterer<double> g_creator_d_##type(                    \
        LayerParameter_LayerType_##type, creator<double>)	

	#define REGISTER_LAYER_CLASS(type, clsname)             ;
	#define REGISTER_LAYER_CLASS_MSC_VER(type, clsname)								\
	  template <typename Dtype>                                                    \
	  Layer<Dtype>* Creator_##clsname(const LayerParameter& param) {               \
		return new clsname<Dtype>(param);                                          \
	  }
	#define REGISTER_LAYER_CREATOR_CLASS_MSC_VER(type, clsname)	\
	  REGISTER_LAYER_CREATOR_MSC_VER(type, Creator_##clsname)

	bool InitLayerFactory();
#endif



// A function to get a specific layer from the specification given in
// LayerParameter. Ideally this would be replaced by a factory pattern,
// but we will leave it this way for now.
// Yangqing's note: With LayerRegistry, we no longer need this thin wrapper any
// more. It is provided here for backward compatibility and should be removed in
// the future.
template <typename Dtype>
Layer<Dtype>* GetLayer(const LayerParameter& param) {
  return LayerRegistry<Dtype>::CreateLayer(param);
}

}  // namespace caffe

#endif  // CAFFE_LAYER_FACTORY_H_
