#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class UnbalancedLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
    UnbalancedLossLayerTest()
      : blob_bottom_a1_(new Blob<Dtype>(5, 2, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(5, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) 
  {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(0);
    filler_param.set_max(1);

    UniformFiller<Dtype> filler(filler_param);

    filler.Fill(this->blob_bottom_a1_);

    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 2;
    }

    blob_bottom_vec_.push_back(blob_bottom_a1_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  
virtual ~UnbalancedLossLayerTest() {
    delete blob_bottom_a1_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }

  void TestForward() {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifiying a weight of 1.
    LayerParameter layer_param;
    UnbalancedLossLayer<Dtype> layer_weight_1(layer_param);
    layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_1 =
        layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Get the loss again with a different objective weight; check that it is
    // scaled appropriately.
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    UnbalancedLossLayer<Dtype> layer_weight_2(layer_param);
    layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_2 =
        layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype kErrorMargin = 1e-5;
    EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);

    // Make sure the loss is non-trivial.
    const Dtype kNonTrivialAbsThresh = 1e-1;
    EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
  }

  Blob<Dtype>* const blob_bottom_a1_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(UnbalancedLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(UnbalancedLossLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(UnbalancedLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  UnbalancedLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(UnbalancedLossLayerTest, PrintBackward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnbalancedLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_a1_->count(); ++i) {
    cout << "bottom data " << i << " " << this->blob_bottom_a1_->cpu_data()[i] << endl;
  }
  
  for (int i = 0; i < this->blob_top_loss_->count(); ++i) {
    cout << "top data " << i << " " << this->blob_top_loss_->cpu_data()[i] << endl;
  }
  for (int i = 0; i < this->blob_top_loss_->count(); ++i) {
    this->blob_top_loss_->mutable_cpu_diff()[i] = i;
  }
  vector<bool> propagate_down;
  propagate_down.push_back(true);
  propagate_down.push_back(true);
  layer.Backward(this->blob_top_vec_,propagate_down, this->blob_bottom_vec_);

  for (int i = 0; i < this->blob_bottom_a1_->count(); ++i) {
    cout << "bottom diff " << i << " " << this->blob_bottom_a1_->cpu_diff()[i] << endl;
  }

}

}  // namespace caffe
