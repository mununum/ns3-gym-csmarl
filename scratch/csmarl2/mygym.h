#ifndef MYGYM_H
#define MYGYM_H

#include "ns3/opengym-module.h"
#include "ns3/wifi-module.h"

namespace ns3 {

class MyGymAgent;

class MyGymEnv : public OpenGymEnv
{
public:
  MyGymEnv (); // for typeid registration
  MyGymEnv (NodeContainer &srcNodes, Time stepTime, std::string algorithm, bool debug);

  virtual ~MyGymEnv ();
  static TypeId GetTypeId (void);
  virtual void DoDispose ();

  Ptr<OpenGymSpace> GetActionSpace ();
  Ptr<OpenGymSpace> GetObservationSpace ();

  bool GetGameOver ();
  Ptr<OpenGymDataContainer> GetObservation ();
  float GetReward ();
  std::string GetExtraInfo ();
  bool ExecuteActions (Ptr<OpenGymDataContainer> actions);

  void PrintResults ();

  static void SrcTxDone (Ptr<MyGymEnv> entity, uint32_t idx, const WifiMacHeader &hdr);
  static void PhyStateChange (Ptr<MyGymEnv> entity, uint32_t idx, Time start, Time duration,
                              WifiPhyState state);

private:
  enum Algorithm { IEEE80211, O_DCF, RL };

  void ScheduleNextStateRead ();
  void StepState ();
  void SetCw (Ptr<Node> node, uint32_t cwalue);
  uint32_t GetCw (Ptr<Node> node);

  uint32_t GetQueueLength (Ptr<Node> node);

  uint32_t m_numAgents;
  std::vector<Ptr<MyGymAgent>> m_agents;

  std::vector<uint32_t> m_obsShape;
  Time m_stepTime;

  Algorithm m_algorithm;

  bool m_debug;

  uint32_t m_perAgentObsDim;

  double m_rewardSum;
  double m_queueLength;
};

} // namespace ns3

#endif