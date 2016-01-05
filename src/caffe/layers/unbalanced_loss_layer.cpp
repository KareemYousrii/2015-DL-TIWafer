#include <vector>

#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe 
{

template <typename Dtype>
	void UnbalancedLossLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{

	LossLayer<Dtype>::Reshape(bottom, top);

	// We use diff0 and diff1 as place holders for bottom[0] and bottom[1]
	// to avoid changing the original data, which is to be used in backprop
	diff_.ReshapeLike(*bottom[0]);
}

	/**
	* @param: bottom[0] is assumed to be the output of the softmax layer. 
	*
	* @param bottom[1] is assumed to be the labels for each of the patches in each
	* image, where a 'good' sample is assumed to be a '0', while a defect is assumed
	* to be a '1'
	*/
template <typename Dtype>
	void UnbalancedLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) 
{
	// The number of samples in the mini-batch
	int num_samples = bottom[0]->shape(0);
	int num_elem = bottom[0]->num();

	caffe_copy(num_elem, bottom[0]->cpu_data(), diff_.mutable_cpu_data());

	// Apply the natural logarithm to the softmax output
	caffe_log(
		num_elem,
		diff_.cpu_data(),
		diff_.mutable_cpu_data());

	Dtype num_good = Dtype(0);
	Dtype num_defect = Dtype(0);
	Dtype sum_good = Dtype(0);
	Dtype sum_defect = Dtype(0);
	for(int i = 0; i < num_samples; i++)
	{
		if(bottom[1]->cpu_data()[i] == 0)
		{
			sum_good -= diff_.cpu_data()[i * 2];	
			num_good++;
		}		
		
		else
		{
			sum_defect -= diff_.cpu_data()[i * 2 + 1];
			num_defect++;
		}
	}	

	// Calculate the final loss
	Dtype loss = (((Dtype(1)/(num_good == 0 ? 1 : num_good)) * sum_good) + ((Dtype(1)/(num_defect == 0 ? 1 : num_defect)) * sum_defect));
	top[0]->mutable_cpu_data()[0] = loss;
}

/* We need to calculate -1/num_good * 1/p(label(P) = good) for the vector
 * containing the probability that a patch is good, and -1/num_defect * 1/p(label(P) = defect)
 * for the vector containing the probabilities that a patch is defective */
template <typename Dtype>
void UnbalancedLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{

	int num_elem = bottom[0] -> num();

	if (propagate_down[0])
	{
		caffe_copy(num_elem, bottom[0]->cpu_data(), bottom[0]->mutable_cpu_diff());

		// Calculate 1/probabilities
		caffe_powx(
			num_elem,
			bottom[0]->cpu_diff(),
			Dtype(-1),
			bottom[0]->mutable_cpu_diff());
	
		bottom[0]->scale_diff(Dtype(-1));
	}
}

#ifdef CPU_ONLY
STUB_GPU(UnbalancedLossLayer);
#endif

INSTANTIATE_CLASS(UnbalancedLossLayer);
REGISTER_LAYER_CLASS(UnbalancedLoss);

}
