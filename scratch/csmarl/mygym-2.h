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
  MyGymEnv2 (NodeContainer agents, Time simTime, Time stepTime, std::string algorithm,
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

private:
  void ScheduleNextStateRead ();
  void StepState ();
  void SetCw (Ptr<Node> node, uint32_t cwValue);
  uint32_t GetCw (Ptr<Node> node);

  uint32_t GetQueueLength (Ptr<Node> node);

  NodeContainer m_agents;
  std::vector<Ptr<MyGymNodeState>> m_agent_state;

  std::vector<uint32_t> m_obs_shape;
  Time m_interval = Seconds (0.005);
  Time m_simTime;

  Algorithm m_algorithm;
  bool m_debug;

  double m_reward_sum;
  double m_reward_indiv_sum;

  float m_queue_reward;
  float m_utility_reward;

  uint32_t m_perAgentObsDim;
};

} // namespace ns3

#endif