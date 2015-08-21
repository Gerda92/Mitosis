#pragma once

#include <string>

#include "Data.h"
//#include "HistogramAggregator.h"
//#include "StatisticsAggregators.h"
#include "Random.h"

#include "Sherwood.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood {

/// <summary>
/// A feature that orders data points using one of their coordinates,
/// i.e. by projecting them onto a coordinate axis.
/// </summary>
class FeatureResponse : IFeatureResponse {

	int nfeatures_;
	int axis_;

	public:

	FeatureResponse() {
		nfeatures_ = 0;
		axis_ = -1;
	}

	FeatureResponse(int nfeatures) {
		nfeatures_ = nfeatures;
		axis_ = -1;
	}

	FeatureResponse(int nfeatures, int axis) {
		nfeatures_ = nfeatures;
		axis_ = axis;
	}

	FeatureResponse CreateRandom(Random& random);

	// IFeatureResponse implementation
	float GetResponse(const IDataPointCollection& data, unsigned int sampleIndex) const;

	std::string ToString() const;

};

class MyAggregator : public IStatisticsAggregator<MyAggregator> {

	private:

	unsigned short bins_[4];
	unsigned int binCount_;

	unsigned int sampleCount_;

	public:

	double Entropy() const;

	MyAggregator();

	MyAggregator(int nClasses);

	float GetProbability(int classIndex) const;

	int BinCount() const {return binCount_; }

	unsigned int SampleCount() const { return sampleCount_; }

	int FindTallestBinIndex() const;

	// IStatisticsAggregator implementation
	void Clear();

	void Aggregate(const IDataPointCollection& data, unsigned int index);

	void Aggregate(const MyAggregator& aggregator);

	MyAggregator DeepClone() const;

};

template<class F>
class TrainingContext : public ITrainingContext<F, MyAggregator> {

	private:
		
	int nClasses_;

	F* featureResponse_;

	public:
		
	TrainingContext(int nClasses, F* featureResponse) {
		nClasses_ = nClasses;
		featureResponse_ = featureResponse;
	}

	private:
	// Implementation of ITrainingContext
	F GetRandomFeature(Random& random) {
		return featureResponse_->CreateRandom(random);
	}

	MyAggregator GetStatisticsAggregator() {
		return MyAggregator(nClasses_);
	}

	double ComputeInformationGain(const MyAggregator& allStatistics, const MyAggregator& leftStatistics, const MyAggregator& rightStatistics) {
		double entropyBefore = allStatistics.Entropy();

		unsigned int nTotalSamples = leftStatistics.SampleCount() + rightStatistics.SampleCount();

		if (nTotalSamples <= 1)
			return 0.0;

		double entropyAfter = (leftStatistics.SampleCount() * leftStatistics.Entropy() + rightStatistics.SampleCount() * rightStatistics.Entropy()) / nTotalSamples;

		return entropyBefore - entropyAfter;
	}

	bool ShouldTerminate(const MyAggregator& parent, const MyAggregator& leftChild, const MyAggregator& rightChild, double gain) {
		return gain < 0.01;
	}

};

template<class F>
void Test(const Forest<F, MyAggregator>& forest, const Data& testData, std::vector<MyAggregator> &result) {
    int nClasses = forest.GetTree(0).GetNode(0).TrainingDataStatistics.BinCount();

    std::vector<std::vector<int> > leafIndicesPerTree;
    forest.Apply(testData, leafIndicesPerTree);

    result = std::vector<MyAggregator>(testData.Count());

    for (int i = 0; i < testData.Count(); i++) {
		// Aggregate statistics for this sample over all leaf nodes reached
		result[i] = MyAggregator(nClasses);
		for (int t = 0; t < forest.TreeCount(); t++) {
			int leafIndex = leafIndicesPerTree[t][i];
			result[i].Aggregate(forest.GetTree(t).GetNode(leafIndex).TrainingDataStatistics);
		}
    }

}

template<class F>
void Test(const Forest<F, MyAggregator>& forest, const Data& testData, std::vector<float> result) {
	std::vector<MyAggregator> resultRaw = Test(forest, testData);
	result = std::vector<float>(testData.Count());
	for (int i = 0; i < testData.Count(); i++) {
		result[i] = resultRaw[i].GetProbability(0);
	}
}

} } }