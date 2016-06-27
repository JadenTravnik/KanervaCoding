

#include <iostream>
#include "feature.h"

using namespace std;


Feature::Feature():
    	    theta_value(0.0),
    	    frequence_value(0.0),
    	    featureWidth_value(0.0),
            action(1)
{

}

Feature Feature::operator= (Feature feature)
{
	setFixed(feature.getState(), feature.getAction());

	return *this;

}

void Feature::setFixed(vector<double> s , int a)
{
	setState(s);
	setAction(a);

    setTheta(0.0);
    setFrequence(0.0);
    setFeatureWidth(0.0);
}



void Feature::setState(vector<double> s){
	state.resize(4);
	for (int i = 0; i < 4; i++)
	    state[i] = s[i];
}

void Feature::setAction(int a){
	action = a;
}

int Feature::getAction(){
	return action;
}

vector<double> Feature::getState(){
	return state;
}

void Feature::setRandomly(){
	setAction(rand() % 5);
	vector<double> temp;
	temp.resize(4);


    temp[0] = rand() % 100 + 1;
    temp[1] = rand() % 100 +1;
    temp[2] = rand() % 48 + 100;
    temp[3] = (rand() % 64195) + 1340;
    setState(temp);
}


bool Feature:: isNeighbor(Feature &feature2){
    int difference = CalculateDiff(feature2);
    if (difference <= 1)
        return true;
    else
        return false;

}


bool Feature::isDifferent(Feature &feature2){
    int difference = CalculateDiff(feature2);
    if (difference <= 0)
        return false;
    else
        return true;
}

unsigned int Feature::CalculateDiff(Feature &feature2){
	unsigned int difference;
	unsigned int actDif = 0;
	unsigned int stateDif = 0;
	unsigned int state_4_Dif = 0;
	if (action == feature2.getAction())
		actDif = 0;
	else
		actDif = 1;

	for (int i = 0; i < 3; i++){
		if (state[i] != feature2.getState()[i])
			stateDif++;
	}
	double local = state[3];
	double given = feature2.getState()[3];
	double big, small;
	if (local >= given){
		big = local;
		small = given;
	}else
	{
		big = given;
		small = local;
	}

	if (big - small <= big * 0.05){
		state_4_Dif = 0;
	}else
		state_4_Dif = 1;
	difference = actDif + stateDif + state_4_Dif;
	return difference;
}

void Feature::print(){
    cout.width(3);
    cout.precision(3);
    cout.setf(ios::showpoint);

    cout<<"[("<<state[0]<<", "<<state[1]<<", "<<state[2]<<", "<<state[3]<<"), "<<action<<"]  "<<endl;

}

void Feature::exportFeature(ofstream &ftfp){

	ftfp<<"[("<<state[0]<<", "<<state[1]<<", "<<state[2]<<", "<<state[3]<<"), "<<action<<"]  "<<endl;

}










