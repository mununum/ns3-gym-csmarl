#ifndef MYGYM_H
#define MYGYM_H

#include "ns3/opengym-module.h"
#include "ns3/wifi-module.h"
#include "utils.h"

namespace ns3 {

class GymAgent;

class GymEnv : public OpenGymEnv
{
public:
  GymEnv (); // for typeid registration
  GymEnv (NodeContainer &srcNodes, const MyConfig &config);

  virtual ~GymEnv ();
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

  static void SrcTxDone (Ptr<GymEnv> entity, uint32_t idx, const WifiMacHeader &hdr);
  static void PhyStateChange (Ptr<GymEnv> entity, uint32_t idx, Time start, Time duration,
                              WifiPhyState state);

private:
  enum Algorithm { IEEE80211, O_DCF, RL };

  void ScheduleNextStateRead ();
  void StepState ();
  void SetCw (Ptr<Node> node, uint32_t cwValue);
  uint32_t GetCw (Ptr<Node> node);
  void SetSourceInterval (Ptr<Node> node, Time interval);
  Time GetSourceInterval (Ptr<Node> node);

  uint32_t GetQueueLength (Ptr<Node> node);

  uint32_t m_numAgents;
  std::vector<Ptr<GymAgent>> m_agents;

  Time m_stepTime;

  Algorithm m_algorithm;

  bool m_debug;

  std::vector<uint32_t> m_obsShape;

  double m_utilityReward;
  double m_queueReward;

  double m_utilityRewardSum;
  double m_queueRewardSum;
};

} // namespace ns3

#endif