#pragma once

// This file defines some IStatisticsAggregator implementations used by the
// example code in Classification.h, DensityEstimation.h, etc. Note we
// represent IStatisticsAggregator instances using simple structs so that all
// tree data can be stored contiguously in a linear array.

#include <math.h>

#include <limits>
#include <vector>

#include "Sherwood.h"

#include "DataPointCollection.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
  struct HistogramAggregator
  {
  private:
    unsigned short bins_[4];
    int binCount_;

    unsigned int sampleCount_;

  public:
    double Entropy() const;

    HistogramAggregator();

    HistogramAggregator(int nClasses);

    float GetProbability(int classIndex) const;

    int BinCount() const {return binCount_; }

    unsigned int SampleCount() const { return sampleCount_; }

    int FindTallestBinIndex() const;

    // IStatisticsAggregator implementation
    void Clear();

    void Aggregate(const IDataPointCollection& data, unsigned int index);

    void Aggregate(const HistogramAggregator& aggregator);

    HistogramAggregator DeepClone() const;
  };

  class GaussianPdf2d
  {
  private:
    double mean_x_, mean_y_;
    double Sigma_11_, Sigma_12_, Sigma_22_; // symmetric 2x2 covariance matrix
    double inv_Sigma_11_, inv_Sigma_12_, inv_Sigma_22_; // symmetric 2x2 inverse covariance matrix
    double det_Sigma_;
    double log_det_Sigma_;

  public:
    GaussianPdf2d() { }

    GaussianPdf2d(double mu_x, double mu_y, double Sigma_11, double Sigma_12, double Sigma_22);

    double MeanX() const
    {
      return mean_x_;
    }

    double MeanY() const
    {
      return mean_y_;
    }

    double VarianceX() const
    {
      return Sigma_11_;
    }

    double VarianceY() const
    {
      return Sigma_22_;
    }

    double CovarianceXY() const
    {
      return Sigma_12_;
    }

    double GetProbability(float x, float y) const;

    double GetNegativeLogProbability(float x, float y) const;

    double Entropy() const;
  };

} } }
