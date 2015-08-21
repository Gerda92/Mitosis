#include "includes.h"

template<typename C>
vector<float> applyClassifier(C classifier, Data data);

template<>
vector<float> applyClassifier(Ptr<SVM> svm, Data data) {
	vector<float> result(data.Count());
	Mat featMat(data.Count(), data.dimension_, CV_32FC1, &data.data_[0]);
	for(int i = 0; i < featMat.rows; i++) {
		Mat sampleMat = featMat(Rect(0, i, featMat.cols, 1));
		result[i] = svm->predict(sampleMat);
	}
	return result;
}

template<>
vector<float> applyClassifier(Forest<FeatureResponse, MyAggregator> forest, Data data) {
	vector<float> result(data.Count());
	Mat featMat(data.Count(), data.dimension_, CV_32FC1, &data.data_[0]);
	vector<MyAggregator> aggregators;

	Test<FeatureResponse>(forest, data, aggregators);
	for(int i = 0; i < featMat.rows; i++) {
		result[i] = aggregators[i].GetProbability(1);
	}
	return result;
}

template<>
vector<float> applyClassifier(double template_matching_threshold, Data data) {
	vector<float> result(data.Count());
	for(int i = 0; i < data.Count(); i++) {
		result[i] = data.GetDataPoint(i)[0];
	}
	return result;
}

void plot(Data data, vector<float> labels, Mat &canvas, bool training = true) {
	
	
	for(int i = 0; i < data.Count(); i++) {
		if (training) {
			circle(canvas, Point(data.GetDataPoint(i)[0]*200,data.GetDataPoint(i)[1]/255.0*200), 4, Scalar(0, 0, 0), -1);
			circle(canvas, Point(data.GetDataPoint(i)[0]*200,data.GetDataPoint(i)[1]/255.0*200), 3, Scalar(255, 255*labels[i], 0), -1);
		} else {
			circle(canvas, Point(data.GetDataPoint(i)[0]*200,data.GetDataPoint(i)[1]/255.0*200), 1, Scalar(255, 255*labels[i], 0), -1);
		}
	}
	
	
}

void trainSVM(Data data, Ptr<SVM> &svm, double c, double gamma) {
	Mat featMat(data.Count(), data.dimension_, CV_32FC1, &data.data_[0]);
	imwrite("featmat.png", featMat);
	Mat labelMat(data.Count(), 1, CV_32SC1, &data.labels_[0]);
	labelMat = labelMat*2-1;

    // Set up SVM's parameters
    SVM::Params params;
    params.svmType = SVM::C_SVC;
	params.kernelType = SVM::RBF;
	params.gamma = gamma;
	//params.kernelType = SVM::LINEAR;
	params.C = c;
	float w[] = {0.1, 0.9};
	Mat weights(2, 1, CV_32FC1, w);
	params.classWeights = weights;
    params.termCrit   = TermCriteria(TermCriteria::MAX_ITER, 1000000, 0.01);

    // Train the SVM
    svm = StatModel::train<SVM>(featMat, ROW_SAMPLE, labelMat, params);

	int count = 0; int positive = 0; int correct_positive = 0;
	for(int i = 0; i < featMat.rows; i++) {
		Mat sampleMat = featMat(Rect(0, i, featMat.cols, 1));
		float result = svm->predict(sampleMat);
		if (data.labels_[i] == 1) {
			positive++; if (result == data.labels_[i]) correct_positive++;
		}
		if (result == data.labels_[i]) count++;
	}

	cout<<"Training accuracy : "<<count<<" / "<<featMat.rows<<" "<<count/featMat.rows<<endl;
	cout<<"Correct positive: "<<correct_positive<<" / "<<positive<<endl;

	std::auto_ptr<Data> testingData = Data::Generate2dGrid(make_pair(0, 1.0), 400, make_pair(0, 255.0), 400);
	
	vector<float> labels = applyClassifier(svm, *testingData);
	Mat canvas(200, 200, CV_8UC3, Scalar(0,0,0));
	plot(*testingData, labels, canvas, false);
	plot(data, vector<float>(data.labels_.begin(), data.labels_.end()), canvas);
	_mkdir("svmm/");
	imwrite("svmm/"+to_string(c)+"_"+to_string(gamma)+".png", canvas);
}

void quickTrain(Data data, string forest_file) {
	
	TrainingParameters trainingParameters;
    trainingParameters.MaxDecisionLevels = 3;
    trainingParameters.NumberOfCandidateFeatures = 20;
    trainingParameters.NumberOfCandidateThresholdsPerFeature = 200;
    trainingParameters.NumberOfTrees = 5;
    trainingParameters.Verbose = true;
	
	Random random;

	FeatureResponse fResp(data.dimension_);
		
	TrainingContext<FeatureResponse> classificationContext(2, &fResp);

	auto_ptr<Forest<FeatureResponse, MyAggregator>> forest
		= ForestTrainer<FeatureResponse, MyAggregator>::TrainForest (
		random, trainingParameters, classificationContext, data);

	(*forest).Serialize(forest_file);
	
	vector<MyAggregator> result;

	Test<FeatureResponse>(*forest, data, result);
	float count = 0;
	for(int i = 0; i < result.size(); i++) {
		//cout<<result[i].GetProbability(1)<<" "<<labels[i]<<endl;
		count += ((result[i].GetProbability(1) >= 0.5? 1 : 0) == data.labels_[i]);
	}
	count /= result.size();
	cout<<"Training accuracy : "<<int(count*result.size())<<" / "<<result.size()<<" "<<count<<endl;
}

void train() {
	float arr[] = {1, 1,  2, 3,  1, 3,  -1, -3,  -5, -1,  1, -1, -2, 4};
	vector<float> feat(arr, arr+14);
	int arrl[] = {0, 0, 0, 1, 1, 1, 1};
	vector<int> lab(arrl, arrl+7);

	Data data(feat, 2, lab);

    TrainingParameters trainingParameters;
    trainingParameters.MaxDecisionLevels = 10;
    trainingParameters.NumberOfCandidateFeatures = 20;
    trainingParameters.NumberOfCandidateThresholdsPerFeature = 3;
    trainingParameters.NumberOfTrees = 10;
    trainingParameters.Verbose = true;

    // Load training data for a 2D density estimation problem.
    //std::auto_ptr<Data> trainingData = std::auto_ptr<Data> (&data);
	
	Random random;

	FeatureResponse fResp(2);
		
	TrainingContext<FeatureResponse> classificationContext(2, &fResp);

	std::auto_ptr<Forest<FeatureResponse, MyAggregator> > forest 
		= ForestTrainer<FeatureResponse, MyAggregator>::TrainForest (
			random, trainingParameters, classificationContext, data);
	
	vector<MyAggregator> result;
	std::auto_ptr<Data> testingData = Data::Generate2dGrid(make_pair(-5.0, 5.0), 400, make_pair(-5.0, 5.0), 400);
	Mat canvas(200, 200, CV_8UC3, Scalar(0,0,0));
	Test<FeatureResponse>(*forest, *testingData, result);

}