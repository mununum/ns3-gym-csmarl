#ifndef MY_GYM_2_H
#define MY_GYM_2_H

#include "ns3/node-container.h"
#include "ns3/wifi-module.h"
#include "ns3/opengym-module.h"

namespace ns3 {

class MyGymNodeState;

class MyGymEnv2 : public OpenGymEnv
{
public:
  enum Algorithm { IEEE80211, O_DCF, RL };

  MyGymEnv2 (); // for typeid registration
  MyGymEnv2 (NodeContainer agents, Time stepTime, std::string algorithm,
             std::map<uint32_t, std::set<uint32_t>> neighbors,
             std::map<uint32_t, uint32_t> degree,
             std::map<uint32_t, double> neiInvDegSum,
             bool debug);

  virtual ~MyGymEnv2 ();
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

  static void SrcTxDone (Ptr<MyGymEnv2> entity, Ptr<Node> node, uint32_t idx,
                         const WifiMacHeader &hdr);
  static void PhyStateChange (Ptr<MyGymEnv2> entity, uint32_t idx, Time start, Time duration, 
                              WifiPhyState state);

private:
  void ScheduleNextStateRead ();
  void StepState ();
  void SetCw (Ptr<Node> node, uint32_t cwValue);
  uint32_t GetCw (Ptr<Node> node);

  uint32_t GetQueueLength (Ptr<Node> node);

  NodeContainer m_agents;
  std::vector<Ptr<MyGymNodeState>> m_agent_state;

  std::vector<uint32_t> m_obs_shape;
  Time m_stepTime = Seconds (0.005);

  Algorithm m_algorithm;
  bool m_debug;

  double m_reward_sum;
  double m_reward_indiv_sum;

  float m_queue_reward;
  float m_utility_reward;

  uint32_t m_perAgentObsDim;

  // for graphical reward calculation
  std::map<uint32_t, std::set<uint32_t>> m_neighbors;
  std::map<uint32_t, uint32_t> m_degree;
  std::map<uint32_t, double> m_neiInvDegSum;

};

} // namespace ns3

#endif