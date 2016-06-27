
#ifndef KANERVACODING_H_
#define KANERVACODING_H_

#include <fstream>
#include "feature.h"
#include <vector>




class TcpLearning
{
public:

	class QLearning
	{
		//// state is a list of state variables: <m_ack_ewma, m_send_ewma, m_rtt_ratio, m_ssThresh>, action is an integer from 0 to 4, a feature is a state-action pair
	private:
		std::map<std::vector<double>, double> q;
		int action; //// may have 5 actions
		double epsilon;
		double alpha;
		double gam;
		bool isFuzzy;

		vector<Feature> features;



	public:
		QLearning();
		double getQ(std::vector<double>state, int action);
		double getQ_fuzzy(std::vector<double>state, int action);
		void learnQ(std::vector<double>state, int action, double reward, double value, std::ofstream &stateActionPair_timeOrder);

		void learn_sarsa(vector<double> last_st, int lastAction, double reward, vector<double>st,
				int action, ofstream &sa_file);

		void learnQ_fuzzy(std::vector<double>state, int action, double reward, double value);
		int chooseAction(std::vector<double>state, std::ofstream &of, std::ofstream &newfile);
		int chooseBestAction(std::vector<double>state, std::ofstream &of, std::ofstream &newfile);
		void learn(std::vector<double>state1,int action1, double reward, std::vector<double>state2, std::ofstream &sa_file);
		void printQ(std::ofstream &of);
		void printStatistics(std::ofstream &sta);
		int getTotalSize();
		void printQChange(double t, std::ofstream &file);
		void setFeatures(vector<Feature> allFeatures);
		void computeFeatureWidth();
		void printTheta(double t, std::ofstream &file);
		void printFeatures(std::ofstream &file);
		void printStateActionPairs(std::ofstream &stateActionPair);
		double get_alpha(){return alpha;}
		void set_alpha(double new_alpha){alpha = new_alpha;}
		double get_epsilon(){return epsilon;}
		void set_epsilon(double new_epsilon){epsilon = new_epsilon;}
		bool getFuzzyBool(){return isFuzzy;}
		void setFuzzyBool(bool new_bool){isFuzzy = new_bool;}
	};


    
    
    
    
    

  TcpLearning (void);
  /**
   * \brief Copy constructor
   * \param sock the object to copy
   */
  TcpLearning (const TcpLearning& sock);
  virtual ~TcpLearning (void);

  // From TcpSocketBase
  virtual int Connect (const Address &address);
  virtual int Listen (void);

protected:
  virtual uint32_t Window (void); // Return the max possible number of unacked bytes
  virtual Ptr<TcpSocketBase> Fork (void); // Call CopyObject<TcpReno> to clone me
  virtual void NewAck (const SequenceNumber32& seq); // Inc cwnd and call NewAck() of parent
  virtual void DupAck (const TcpHeader& t, uint32_t count);  // Fast retransmit
  virtual void Retransmit (void); // Retransmit timeout
  virtual void ReceivedAck (Ptr<Packet> packet, const TcpHeader& tcpHeader); // Process received ACK


  virtual void     SetSegSize (uint32_t size);
  virtual void     SetInitialSSThresh (uint32_t threshold);
  virtual uint32_t GetInitialSSThresh (void) const;
  virtual void     SetInitialCwnd (uint32_t cwnd);
  virtual uint32_t GetInitialCwnd (void) const;


  virtual void GetAckEwma (void);
  virtual void GetSendEwma (const TcpHeader& tcpHeader);
  virtual void GetRttRatio (const TcpHeader& tcpHeader);
  virtual void GetUtility (void);
  virtual void GetLearning(void);
  virtual void WarmupLearning();
  virtual TracedValue<uint32_t> AdjustWindow();


  void GenerateFeatureVector();
  void GenerateFeatureVectorFromFile();
  void RecordFeatureVector_simple();
  void setLearnerFeatures();
private:
  /**
   * \brief Set the congestion window when connection starts
   */
  void InitializeCwnd (void);

protected:
  TracedValue<uint32_t>  m_cWnd;         //!< Congestion window
  TracedValue<uint32_t>  m_ssThresh;     //!< Slow Start Threshold, one state variable
  uint32_t               m_initialCWnd;  //!< Initial cWnd value
  uint32_t               m_initialSsThresh;  //!< Initial Slow Start Threshold value

  SequenceNumber32       m_recover;      //!< Previous highest Tx seqnum for fast recovery
  uint32_t               m_retxThresh;   //!< Fast Retransmit threshold
  bool                   m_inFastRec;    //!< currently in fast recovery
  bool                   m_limitedTx;    //!< perform limited transmit ////

  // TCP Learning
  uint32_t               m_ack_ewma;   //// one state variable

  uint32_t               m_send_ewma;   //// one state variable

  double                 m_rtt_ratio;   //// one state variable


  double                 m_baseRtt;
  Time                   m_lastSample;
  double                 m_interval;
  double                 m_dataSentLastInterval;
  double                 m_totalRTTLastInterval;
  double                 m_samplesTake;
  double                 m_sampleThroughput;
  double                 m_sampleRTT;
  double                 m_para;
  double                 m_utility;
  double                 m_pre_utility;
//  double                 epsilon;
  int                    n_action;


  std::ofstream          output_file;
  std::ofstream          newfile;


  vector<Feature> features_1;
  vector<int> actionList;

  /////////////// learning class ////////////////////
  QLearning ai;
  int lastAction;
  std::vector<double> st;
  std::vector<double> last_st;
  bool firstTime;
  bool actionFailed;
  int counter;

};



#endif
