#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>

#include "Sherwood.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{

  /// <summary>
  /// A collection of data points, each represented by a float[] and (optionally)
  /// associated with a string class label and/or a float target value.
  /// </summary>
  class Data: public IDataPointCollection
  {

  public:

    std::vector<float> data_;
    int dimension_;

    // only for classified data...
    std::vector<int> labels_;

    std::map<std::string, int> labelIndices_; // map string labels to integers

    // only for regression problems...
    std::vector<float> targets_;

    static const int UnknownClassLabel = -1;

	Data() {}

	Data(std::vector<float> data, int dim, std::vector<int> labels) {
		data_ = data; dimension_ = dim; labels_ = labels;
	}

    /// <summary>
    /// Generate a 2D dataset with data points distributed in a grid pattern.
    /// Intended for generating visualization images.
    /// </summary>
    /// <param name="rangeX">x-axis range</param>
    /// <param name="nStepsX">Number of grid points in x direction</param>
    /// <param name="rangeY">y-axis range</param>
    /// <param name="nStepsY">Number of grid points in y direction</param>
    /// <returns>A new Data</returns>
    static  std::auto_ptr<Data> Generate2dGrid(
      std::pair<float, float> rangeX, int nStepsX,
      std::pair<float, float> rangeY, int nStepsY);

	bool HasLabels() const
    {
      return labels_.size() != 0;
    }

    unsigned int Count() const
    {
      return data_.size()/dimension_;
    }

	// Needed for FeatureResponse
    const float* GetDataPoint(int i) const
    {
      return &data_[i*dimension_];
    }

	// Needed for HistogramAggregator
    int GetIntegerLabel(int i) const
    {
      if (!HasLabels())
        throw std::runtime_error("Data have no associated class labels.");

      return labels_[i]; // may throw an exception if index is out of range
    }
  };

} } }