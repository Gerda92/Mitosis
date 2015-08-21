#include "RandomForest.h"

#include <sstream>

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood {

	FeatureResponse FeatureResponse::CreateRandom(Random& random) {
		return FeatureResponse(nfeatures_, random.Next(0, nfeatures_));
	}

	float FeatureResponse::GetResponse(const IDataPointCollection& data, unsigned int sampleIndex) const {
		const Data& concreteData = (Data&)(data);
		return concreteData.GetDataPoint((int)sampleIndex)[axis_];
	}

	std::string FeatureResponse::ToString() const {
		std::stringstream s;
		s << "FeatureResponse(" << axis_ << ")";

		return s.str();
	}

	double MyAggregator::Entropy() const {
		if (sampleCount_ == 0)
			return 0.0;

		double result = 0.0;
		for (int b = 0; b < BinCount(); b++)
		{
			double p = (double)bins_[b] / (double)sampleCount_;
			result -= p == 0.0 ? 0.0 : p * log(p)/log(2.0);
		}

		return result;
	}

	MyAggregator::MyAggregator() {
		binCount_ = 0;
		sampleCount_ = 0;
	}

	MyAggregator::MyAggregator(int nClasses) {
		if(nClasses>4)
			throw std::runtime_error("HistogramAggregator supports a maximum of four classes.");
		binCount_ = nClasses;
		for(int b=0; b<binCount_; b++)
			bins_[b] = 0;
		sampleCount_ = 0;
	}

	float MyAggregator::GetProbability(int classIndex) const {
		return (float)(bins_[classIndex]) / sampleCount_;
	}

	int MyAggregator::FindTallestBinIndex() const {
		unsigned int maxCount = bins_[0];
		int tallestBinIndex = 0;

		for (int i = 1; i < BinCount(); i++) {
			if (bins_[i] > maxCount) {
				maxCount = bins_[i];
				tallestBinIndex = i;
			}
		}

		return tallestBinIndex;
	}

	// IStatisticsAggregator implementation
	void MyAggregator::Clear() {
		for (int b = 0; b < BinCount(); b++)
			bins_[b] = 0;

		sampleCount_ = 0;
	}

	void MyAggregator::Aggregate(const IDataPointCollection& data, unsigned int index) {
		const Data& concreteData = (const Data&)(data);

		bins_[concreteData.GetIntegerLabel((int)index)]++;
		sampleCount_ += 1;
	}

	void MyAggregator::Aggregate(const MyAggregator& aggregator) {
		assert(aggregator.BinCount() == BinCount());

		for (int b = 0; b < BinCount(); b++)
			bins_[b] += aggregator.bins_[b];

		sampleCount_ += aggregator.sampleCount_;
	}

	MyAggregator MyAggregator::DeepClone() const {
		MyAggregator result(BinCount());

		for (int b = 0; b < BinCount(); b++)
			result.bins_[b] = bins_[b];

		result.sampleCount_ = sampleCount_;

		return result;
	}

} } }