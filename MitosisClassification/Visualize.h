#include <stdexcept>
#include <algorithm>

#include "Graphics.h"

#include "Sherwood.h"

#include "StatisticsAggregators.h"
#include "FeatureResponseFunctions.h"
#include "Data.h"
#include "Classification.h"
#include "PlotCanvas.h"

using namespace MicrosoftResearch::Cambridge::Sherwood;

template<class F>

std::auto_ptr<Bitmap<PixelBgr> > VisualizeGerda(
      Forest<F, HistogramAggregator>& forest,
      Data& trainingData,
      MicrosoftResearch::Cambridge::Sherwood::Size PlotSize,
      PointF PlotDilation) // where F: IFeatureResponse
    {
      // Size PlotSize = new Size(300, 300), PointF PlotDilation = new PointF(0.1f, 0.1f)
      // Generate some test samples in a grid pattern (a useful basis for creating visualization images)
      PlotCanvas plotCanvas(trainingData.GetRange(0), trainingData.GetRange(1), PlotSize, PlotDilation);

      std::auto_ptr<Data> testData = std::auto_ptr<Data>(
        Data::Generate2dGrid(plotCanvas.plotRangeX, PlotSize.Width, plotCanvas.plotRangeY, PlotSize.Height) );

      std::cout << "\nApplying the forest to test data..." << std::endl;

      std::vector<std::vector<int> > leafNodeIndices;
      forest.Apply(*testData, leafNodeIndices);

      // Same colours as those used in the book
      assert(trainingData.CountClasses()<=4);
      PixelBgr colors[4];
      colors[0] = PixelBgr::FromArgb(183, 170, 8);
      colors[1] = PixelBgr::FromArgb(194, 32, 14);
      colors[2] = PixelBgr::FromArgb(4, 154, 10);
      colors[3] = PixelBgr::FromArgb(13, 26, 188);

      PixelBgr grey = PixelBgr::FromArgb(127, 127, 127);

      // Create a visualization image
      std::auto_ptr<Bitmap<PixelBgr> > result = std::auto_ptr<Bitmap<PixelBgr> >(
        new Bitmap<PixelBgr>(PlotSize.Width, PlotSize.Height) );

      // For each pixel...
      int index = 0;
      for (int j = 0; j < PlotSize.Height; j++)
      {
        for (int i = 0; i < PlotSize.Width; i++)
        {
          // Aggregate statistics for this sample over all leaf nodes reached
          HistogramAggregator h(trainingData.CountClasses());
          for (int t = 0; t < forest.TreeCount(); t++)
          {
            int leafIndex = leafNodeIndices[t][index];
            h.Aggregate(forest.GetTree((t)).GetNode(leafIndex).TrainingDataStatistics);
          }

          // Let's muddy the colors with grey where the entropy is high.
          float mudiness = 0.5f*(float)(h.Entropy());

          float R = 0.0f, G = 0.0f, B = 0.0f;

          for (int b = 0; b < trainingData.CountClasses(); b++)
          {
            float p = (1.0f-mudiness)*h.GetProbability(b); // NB probabilities sum to 1.0 over the classes

            R += colors[b].R * p;
            G += colors[b].G * p;
            B += colors[b].B * p;
          }

          R += grey.R * mudiness;
          G += grey.G * mudiness;
          B += grey.B * mudiness;

          PixelBgr c = PixelBgr::FromArgb((unsigned char)(R), (unsigned char)(G), (unsigned char)(B));

          result->SetPixel(i, j, c); // painfully slow but safe

          index++;
        }
      }
      Graphics<PixelBgr> g(result->GetBuffer(), result->GetWidth(), result->GetHeight(), result->GetStride());

      for (unsigned int s = 0; s < trainingData.Count(); s++)
      {
        PointF x(
          (trainingData.GetDataPoint(s)[0] - plotCanvas.plotRangeX.first) / plotCanvas.stepX,
          (trainingData.GetDataPoint(s)[1] - plotCanvas.plotRangeY.first) / plotCanvas.stepY);

        RectangleF rectangle(x.X - 3.0f, x.Y - 3.0f, 6.0f, 6.0f);
        g.FillRectangle(colors[trainingData.GetIntegerLabel(s)], rectangle.X, rectangle.Y, rectangle.Width, rectangle.Height);
        g.DrawRectangle(PixelBgr::FromArgb(0,0,0), rectangle.X, rectangle.Y, rectangle.Width, rectangle.Height);
      }

      return result;
    }