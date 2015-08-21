#include "Data.h"

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{

  /// <summary>
  /// Generate a 2D dataset with data points distributed in a grid pattern.
  /// Intended for generating visualization images.
  /// </summary>
  /// <param name="rangeX">x-axis range</param>
  /// <param name="nStepsX">Number of grid points in x direction</param>
  /// <param name="rangeY">y-axis range</param>
  /// <param name="nStepsY">Number of grid points in y direction</param>
  /// <returns>A new Data</returns>

  std::auto_ptr<Data> Data::Generate2dGrid(
    std::pair<float, float> rangeX, int nStepsX,
    std::pair<float, float> rangeY, int nStepsY)
  {

    if (rangeX.first >= rangeX.second)
      throw std::runtime_error("Invalid x-axis range.");
    if (rangeY.first >= rangeY.second)
      throw std::runtime_error("Invalid y-axis range.");

    std::auto_ptr<Data> result =  std::auto_ptr<Data>(new Data());

    result->dimension_ = 2;

    float stepX = (rangeX.second - rangeX.first) / nStepsX;
    float stepY = (rangeY.second - rangeY.first) / nStepsY;

    for (int j = 0; j < nStepsY; j++)
      for (int i = 0; i < nStepsX; i++)
      {
        result->data_.push_back(rangeX.first + i * stepX);
        result->data_.push_back(rangeY.first + j * stepY);
      }

      return result;
  }

} } }
