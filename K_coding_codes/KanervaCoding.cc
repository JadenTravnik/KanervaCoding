//// Use Kanerva coding to learn congestion window in network domain
#include <string.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <map>
#include <time.h>
#include <assert.h>
#include <sstream>



#define NumActions 5
#define NumFeatures 102


#define AlphaFactor 0.99995
#define EpsilonFactor 0.9995

#define storePath "/Users/weili/Desktop/learning/"

using namespace std;



TcpLearning::TcpLearning(void):
m_retxThresh(3),
m_inFastRec(false),
m_limitedTx (false), // mute valgrind, actual value set by the attribute system

m_ack_ewma(0),    ////
m_send_ewma(0),   ////
m_rtt_ratio(0.0), ////
m_baseRtt(0.0),

m_lastSample(0),
m_interval(0.1),
m_dataSentLastInterval(0),
m_totalRTTLastInterval(0),
m_samplesTake(0),
m_sampleThroughput(0),
m_sampleRTT(0),



m_para(1),
m_utility(-999999),
m_pre_utility(-999999),

lastAction(2),
firstTime(true),
actionFailed(false),
counter(0)
{
    
    
    string fileName_output_file_Q_table = "output_file_Q_table.txt";
    string output_file_Path = storePath + fileName_output_file_Q_table;
    output_file.open(output_file_Path.c_str(), ios::out );
    if (!output_file.is_open())
        std::cout << "output_file_Q_table.txt does not open" << std::endl;
    else
        output_file << "\n***************************************************"<<std::endl;
    
    
    string fileName_newfile = "newfile.txt";
    string newfile_Path = storePath + fileName_newfile;
    newfile.open(newfile_Path.c_str(), ios::out );
    if (!newfile.is_open())
        std::cout << "newfile.txt does not open" << std::endl;
    else
        newfile << "\n***************************************************"<<std::endl;
    
    actionList.resize(NumActions);
    actionList[0] = -1;
    actionList[1] = 0;
    actionList[2] = 5;
    actionList[3] = 10;
    actionList[4] = 20;

    
    st.resize(4);
    last_st.resize(4);
    
    features_1.resize(NumFeatures);
    
    
    GenerateFeatureVector();              //// create features randomly
    ////GenerateFeatureVectorFromFile();  //// create features from file
    setLearnerFeatures();
}


TcpLearning::~TcpLearning(void)
{
}


void TcpLearning::GenerateFeatureVectorFromFile()
// Generate the features from file
{

    FILE *filefeature;
    string subPath = "foo_folder/";
    string filePath = storePath + subPath;
    string filefeature_fileName = "foo.txt";
    string filefeature_fileNameFull = filePath + filefeature_fileName;
    filefeature = fopen(filefeature_fileNameFull.c_str(), "r");

    cout << "= Feature Generation from file " << NumFeatures << endl;
    double ack, send, rtt, threshold;
    double actionValue;
    for (int fIndex = 0; fIndex < NumFeatures; fIndex++)
    {
        fscanf(filefeature, " %lf %lf %lf %lf %lf ", &ack, &send, &rtt, &threshold, &actionValue);
        vector<double>s;
        s.resize(4);
        s[0] = ack;
        s[1] = send;
        s[2] = rtt;
        s[3] = threshold;
        features_1[fIndex].setFixed(s, actionValue);     }

}


void TcpLearning::GenerateFeatureVector() {
// Generate the features randomly
	std::cout << "= Feature Generation (randomly distribution) " << NumFeatures << std::endl;
	int fDiff;
	for (int fIndex = 0; fIndex < NumFeatures; fIndex++) {
		fDiff = 0;
		(features_1[fIndex]).setRandomly();
		while (fDiff < fIndex) //// check the uniqueness of newly randomly generated feature
		{
			if ((features_1[fIndex]).isDifferent(features_1[fDiff]))
				fDiff++;
			else {
				(features_1[fIndex]).setRandomly();
				fDiff = 0;
			}
		}
	}

}



void TcpLearning::setLearnerFeatures() {
	ai.setFeatures(features_1);
	ai.computeFeatureWidth(); //// must recall this once the set of features has changed
}



//// GetLearning is called every time step
void TcpLearning::GetLearning(void) {
        st[0] = m_ack_ewma;        //// which is updated every time step by GetAckEwma()
        st[1] = m_send_ewma;       //// which is updated every time step by GetSendEwma()
        st[2] = m_rtt_ratio;       //// which is updated every time step by GetRttRatio()
        st[3] = m_ssThresh;        //// which is updated every time step by network
    
        if (Simulator::Now().GetSeconds() - m_lastSample.GetSeconds() >= 1) { //// update learning parameters
            
            ai.set_alpha(ai.get_alpha() * AlphaFactor);
            ai.set_epsilon(ai.get_epsilon() * EpsilonFactor);
            m_lastSample = Simulator::Now();
        }
        
        double reward = 0;
        
        if (!firstTime)
        {
            
            if (m_utility <= m_pre_utility) {//// m_utility is measured from network
                reward = -2;
                
            } else
            {
                reward = 2;
                
            }
            
            ai.learn(last_st, lastAction, reward, st, stateActionPair_timeOrder);
            
        }

    
        last_st = st;
    
        //// lastAction is the learned action that changes the congestion window in the network
        if (firstTime)
            lastAction = 2;
        else {
            
            lastAction = ai.chooseAction(last_st, output_file, newfile);
            
        }
    
        firstTime = false;
    
}

    
    
void TcpLearning::GetRttRatio(const TcpHeader& tcpHeader) {
    //// get m_rtt_ratio from network environment
    }
    


void TcpLearning::GetAckEwma() {
    //// get m_ack_ewma from network environment
    }



void TcpLearning::GetSendEwma(const TcpHeader& tcpHeader){
    //// get m_send_ewma from network environment
    }







TcpLearning::QLearning::QLearning() :
		action(2),
		epsilon(0.2),
        alpha(0.9),
        gam(0.9),
        isFuzzy(true) {

}



void TcpLearning::QLearning::setFeatures(vector<Feature> allFeatures)
{
	features = allFeatures;
}


void TcpLearning::QLearning::computeFeatureWidth()
{
	int featureSize = features.size();
	int FeatureDiff[featureSize - 1];
	int idx, sum;
	float sum2, mean, variance;
	int numFeatures = features.size();
	for (int j = 0; j < numFeatures; j++) {
		idx = 0;
		sum = 0;
		sum2 = 0;
		mean = 0;
		variance = 0;
		for (int k = 0; k < numFeatures; k++)
			if (j != k) {
				FeatureDiff[idx] = features[j].CalculateDiff(features[k]);
				idx++;
			}
		for (int m = 0; m < numFeatures - 1; m++)
			sum += FeatureDiff[m];
		mean = sum / (numFeatures - 1);
		for (int n = 0; n < numFeatures - 1; n++)
			sum2 += (FeatureDiff[n] - mean) * (FeatureDiff[n] - mean);
		variance = sum2 / (numFeatures - 1 - 1); ////use N-1 instead of N, a correction when your data is only a sample
		features[j].setFeatureWidth(variance);
		cout<<"feature width of feature "<<j<<" is = "<<variance<<endl;
	}

}

double TcpLearning::QLearning::getQ(std::vector<double> state, int action) {
	std::vector<double> sa(state);
	sa.push_back((double) action); //// sa is the state-action pair
	std::map<std::vector<double>, double>::iterator it;
	it = q.find(sa);
	if (it == q.end())
		return 0.0;
	else
		return it->second;
}

double TcpLearning::QLearning::getQ_fuzzy(std::vector<double> state, int action) {
	double thetaSum;
	double membershipGrade;
	int tmpFeatureDiff;

	Feature feature;
	feature.setFixed(state, action);
	for (int ff = 0; ff < NumFeatures; ff++) {
		tmpFeatureDiff = feature.CalculateDiff(features[ff]);
		membershipGrade = (float) exp(
				-(tmpFeatureDiff * tmpFeatureDiff)
						/ (2 * features[ff].getFeatureWidth()));
		thetaSum += features[ff].getTheta() * membershipGrade;

	}
	return thetaSum;

}

int TcpLearning::QLearning::getTotalSize() {
	return q.size();

}




void TcpLearning::QLearning::learn(std::vector<double> state1, int action1,
                                   double reward, std::vector<double> state2, std::ofstream &sa_file) {
    //// state2 is current state, state1 is previous state
    double maxQ = -999999999999999999;
    double maxQ_copy = maxQ;
    double tempQ = 0;
    for (int a = 0; a < NumActions; a++) {
        if (isFuzzy)
            tempQ = getQ_fuzzy(state2, a);
        else
            tempQ = getQ(state2, a);
        
        if (maxQ < tempQ)
            maxQ = tempQ;
    }
    assert(maxQ != maxQ_copy);
    if (isFuzzy)
        learnQ_fuzzy(state1, action1, reward, reward + gam * maxQ);
    else
        learnQ(state1, action1, reward, reward + gam * maxQ, sa_file);
}



void TcpLearning::QLearning::learnQ(std::vector<double> state, int action,
		double reward, double value, std::ofstream &stateActionPair_timeOrder) {
	std::vector<double> sa(state);
	sa.push_back((double) action); //// sa is the state-action pair
	std::map<std::vector<double>, double>::iterator it;
	it = q.find(sa);
	if (it == q.end()) { //// if cannot find this state action pair in the q table, then add this newly encountered state action pair to the q table

		stateActionPair_timeOrder << sa[0] << "  " << sa[1] << "  " << sa[2] << "  " << sa[3] << "  " << sa[4] << std::endl;

		q[sa] = reward;
		freq[sa] = 1;
	} else {
		q[sa] = it->second + alpha * (value - it->second);
    
	}
}

void TcpLearning::QLearning::learnQ_fuzzy(std::vector<double> state, int action,
		double reward, double value) {
	double membershipGrade;
	int tmpFeatureDiff;
	double preQ = getQ_fuzzy(state, action);
	double delta = value - preQ;

	Feature feature;
	feature.setFixed(state, action);
	for (int ff = 0; ff < NumFeatures; ff++) {
		tmpFeatureDiff = feature.CalculateDiff(features[ff]);
		membershipGrade = (float) exp(-(tmpFeatureDiff * tmpFeatureDiff) / (2 * features[ff].getFeatureWidth()));
		features[ff].setTheta(features[ff].getTheta() + alpha * delta * membershipGrade / (double) (NumFeatures) );
	}

}

int TcpLearning::QLearning::chooseAction(std::vector<double> state,
		std::ofstream &of, std::ofstream &newfile) {
	int action;
	if ((double) rand() / RAND_MAX < epsilon) {
		action = rand() % NumActions;
		return action;
	} else {
		action = chooseBestAction(state, of, newfile);
		return action;
	}

}

int TcpLearning::QLearning::chooseBestAction(std::vector<double> state,
		std::ofstream &of, std::ofstream &newfile) {
	int action = 0;
	double maxQ = -999999999999999999;
	double maxQ_copy = maxQ;
	std::vector<int> bestV;

	std::vector<double> q_list;
	q_list.clear();
	for (int a = 0; a < NumActions; a++) {
		double qValue;
		if (isFuzzy)
		    qValue = getQ_fuzzy(state, a);
		else
			qValue = getQ(state, a);

		q_list.push_back(qValue);
	}
	for (int i = 0; i < (int) q_list.size(); i++) {
		if (maxQ < q_list[i]) {
			maxQ = q_list[i];

			bestV.clear();
		}
		if (maxQ == q_list[i])
			bestV.push_back(i);
	}
	assert(maxQ != maxQ_copy);

	if (bestV.size() > 1)
	{
		int index = rand() % bestV.size();
		action = bestV[index];

		if (bestV.size() != NumActions)
			newfile << " multi-actions, randomly get " << action << "\n" << std::endl;
	} else {
		action = bestV[0];
    }

	return action;
}




void TcpLearning::QLearning::printFeatures(std::ofstream &file){
	for (int ft = 0; ft < NumFeatures; ft++) {
		file<<"[("<<features[ft].getState()[0]<<", "<<features[ft].getState()[1]<<", "
				<<features[ft].getState()[2]<<", "<<features[ft].getState()[3]<<"), "
				<<features[ft].getAction()<<"] "<<std::endl;
	}
	file<<"\n"<<std::endl;
	file.close();
}


