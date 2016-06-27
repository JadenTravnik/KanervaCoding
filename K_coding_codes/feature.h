

#ifndef FEATURE_H_
#define FEATURE_H_

#include <vector>

using namespace std;



class Feature
{
public:
    Feature();
    Feature operator=(Feature feature);
    void setRandomly();
    void setFixed(vector<double> s , int a);

    void print();

    void exportFeature(ofstream &ftfp);
    void setState(vector<double> s);
    void setAction(int a);

    vector<double> getState();
    int getAction();

    bool isNeighbor(Feature &feature2);
    bool isDifferent(Feature &feature2);
    unsigned int CalculateDiff(Feature &feature2);


    void setTheta(float theta){ theta_value = theta;}
    float getTheta(){return theta_value;}

    void setFrequence(float frequence){frequence_value = frequence;}
    float getFrequence(){return frequence_value;}

    void setFeatureWidth(float featureWidth){featureWidth_value = featureWidth;}
    float getFeatureWidth(){return featureWidth_value;}


private:
    float theta_value;               //// theta value associted with each prototype(feature)
    float frequence_value;
    float featureWidth_value;

    int action;
    vector<double> state;


};





#endif /* FEATURE_H_ */
