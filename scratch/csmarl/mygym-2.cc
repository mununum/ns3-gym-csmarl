#include "mygym-2.h"
#include "ns3/object.h"
#include "ns3/core-module.h"
#include "ns3/wifi-module.h"
#include "ns3/node-list.h"
#include "ns3/log.h"
#include "ns3/delay-jitter-estimation-2.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("MyGymEnv2");

NS_OBJECT_ENSURE_REGISTERED (MyGymEnv2);

class MyGymNodeState : public Object
{
  friend class MyGymEnv2;

public:
  MyGymNodeState ()
      : m_txPktNum (0),
        m_txPktNumLastVal (0),
        m_queueLength (0),
        m_delaySum (Seconds (0.0)),
        m_delayEwma (0),
        m_delay_estimator (CreateObject<DelayJitterEstimation2> (1))
  {
  }

  void
  Step ()
  {
    m_txPktNumLastVal = m_txPktNum;
  }

private:
  uint64_t m_txPktNum;
  uint64_t m_txPktNumLastVal;

  uint32_t m_queueLength;

  Time m_delaySum;
  double m_delayEwma;

  Ptr<DelayJitterEstimation2> m_delay_estimator; // for MAQ delay
};

MyGymEnv2::MyGymEnv2 ()
{
  NS_LOG_FUNCTION (this);
}

MyGymEnv2::MyGymEnv2 (NodeContainer agents, Time simTime, Time stepTime, std::string algorithm,
                      bool debug = false)
{
  NS_LOG_FUNCTION (this);
  m_agents = agents;
  m_simTime = simTime;
  m_interval = stepTime;

  if (algorithm == "80211")
    m_algorithm = IEEE80211;
  else if (algorithm == "odcf")
    m_algorithm = O_DCF;
  else if (algorithm == "rl")
    m_algorithm = RL;
  else
    NS_ASSERT (false);

  m_debug = debug;

  m_reward_sum = 0.0;

  uint32_t numAgents = m_agents.GetN ();
  for (uint32_t i = 0; i < numAgents; i++)
    {
      m_agent_state.push_back (CreateObject<MyGymNodeState> ());
    }

  m_perAgentObsDim = 5; // Throughput, QueueLength, e2e Latency, Loss%, CW

  Simulator::Schedule (Seconds (0.0), &MyGymEnv2::ScheduleNextStateRead, this);
}

void
MyGymEnv2::ScheduleNextStateRead ()
{
  NS_LOG_FUNCTION (this);

  // notify to python-end only when enabled
  if (m_algorithm == RL)
    Notify ();
  else
    {
      GetGameOver ();
      GetObservation ();
      GetReward ();
      GetExtraInfo ();
    }

  StepState ();

  Simulator::Schedule (m_interval, &MyGymEnv2::ScheduleNextStateRead, this);
}

void
MyGymEnv2::StepState ()
{
  uint32_t numAgents = m_agents.GetN ();

  for (uint32_t i = 0; i < numAgents; i++)
    {
      Ptr<Node> node = m_agents.Get (i);
      Ptr<NetDevice> dev = node->GetDevice (0);
      Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice> (dev);
      Ptr<WifiRemoteStationManager> rman = wifi_dev->GetRemoteStationManager ();

      // reset every statistics
      m_agent_state[i]->Step ();
      rman->GetAggInfo ().Reset ();
    }
}

MyGymEnv2::~MyGymEnv2 ()
{
  NS_LOG_FUNCTION (this);
}

TypeId
MyGymEnv2::GetTypeId (void)
{
  static TypeId tid = TypeId ("MyGymEnv2")
                          .SetParent<OpenGymEnv> ()
                          .SetGroupName ("OpenGym")
                          .AddConstructor<MyGymEnv2> ();
  return tid;
}

void
MyGymEnv2::DoDispose ()
{
  NS_LOG_FUNCTION (this);
}

Ptr<OpenGymSpace>
MyGymEnv2::GetActionSpace ()
{
  NS_LOG_FUNCTION (this);
  uint32_t numAgents = m_agents.GetN ();

  int n_actions = 9;
  Ptr<OpenGymTupleSpace> space = CreateObject<OpenGymTupleSpace> ();

  for (uint32_t i = 0; i < numAgents; i++)
    {
      space->Add (CreateObject<OpenGymDiscreteSpace> (n_actions));
    }

  NS_LOG_DEBUG ("GetActionSpace: " << space);
  return space;
}

Ptr<OpenGymSpace>
MyGymEnv2::GetObservationSpace ()
{
  NS_LOG_FUNCTION (this);
  uint32_t numAgents = m_agents.GetN ();

  m_obs_shape = {numAgents, m_perAgentObsDim};

  float low = 0;
  float high = 10000.0;

  std::string dtype = TypeNameGet<float> ();
  Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace> (low, high, m_obs_shape, dtype);

  NS_LOG_DEBUG ("GetObservationSpace: " << space);
  return space;
}

bool
MyGymEnv2::GetGameOver ()
{
  NS_LOG_FUNCTION (this);
  bool isGameOver = false;
  NS_LOG_DEBUG ("MyGetGameOver: " << isGameOver);
  return isGameOver;
}

uint32_t
MyGymEnv2::GetQueueLength (Ptr<Node> node)
{
  Ptr<NetDevice> dev = node->GetDevice (0);
  Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice> (dev);
  Ptr<WifiMac> wifi_mac = wifi_dev->GetMac ();
  if (m_algorithm == IEEE80211)
    {
      // 80211: Get IFQ length
      Ptr<RegularWifiMac> rmac = DynamicCast<RegularWifiMac> (wifi_mac);
      PointerValue ptr;
      rmac->GetAttribute ("Txop", ptr);
      Ptr<Txop> txop = ptr.Get<Txop> ();
      Ptr<WifiMacQueue> queue = txop->GetWifiMacQueue ();
      return queue->GetNPackets ();
    }
  else
    {
      // O-DCF/RL: Get MAQ length
      Ptr<ODcfAdhocWifiMac> odcf_mac = DynamicCast<ODcfAdhocWifiMac> (wifi_mac);
      PointerValue ptr;
      odcf_mac->GetAttribute ("ODcf", ptr);
      Ptr<ODcf> odcf = ptr.Get<ODcf> ();
      return odcf->GetMAQLength ();
    }
}

Ptr<OpenGymDataContainer>
MyGymEnv2::GetObservation ()
{
  NS_LOG_FUNCTION (this);
  Ptr<OpenGymBoxContainer<float>> box = CreateObject<OpenGymBoxContainer<float>> (m_obs_shape);

  uint32_t numAgents = m_agents.GetN ();
  for (uint32_t i = 0; i < numAgents; i++)
    {
      Ptr<Node> node = m_agents.Get (i);

      // Throughput
      double thpt = m_agent_state[i]->m_txPktNum - m_agent_state[i]->m_txPktNumLastVal;
      thpt /= 1000;

      // QueueLength
      double qlen = GetQueueLength (node);
      m_agent_state[i]->m_queueLength = qlen;

      // Latency
      double lat = m_agent_state[i]->m_delayEwma;
      lat /= 1e9;  // latency unit: sec

      // Loss%
      Ptr<NetDevice> dev = node->GetDevice (0);
      Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice> (dev);
      Ptr<WifiRemoteStationManager> rman = wifi_dev->GetRemoteStationManager ();
      double err_rate = rman->GetAggInfo ().GetFrameErrorRate ();

      // CW
      Ptr<WifiMac> wifi_mac = wifi_dev->GetMac ();
      Ptr<RegularWifiMac> rmac = DynamicCast<RegularWifiMac> (wifi_mac);
      PointerValue ptr;
      rmac->GetAttribute ("Txop", ptr);
      Ptr<Txop> txop = ptr.Get<Txop> ();
      double mincw = txop->GetMinCw ();

      // put it in a box
      box->AddValue (thpt);
      box->AddValue (qlen);
      box->AddValue (lat);
      box->AddValue (err_rate);
      box->AddValue (mincw);

      if (m_debug)
        std::cout << qlen << "\t";
    }

  if (m_debug)
    std::cout << std::endl;

  return box;
}

float
MyGymEnv2::GetReward ()
{
  NS_LOG_FUNCTION (this);

  const double reward_scale = 1e5;

  uint32_t numAgents = m_agents.GetN ();
  float reward = 0.0;

  for (uint32_t i = 0; i < numAgents; i++)
    {
      reward -= m_agent_state[i]->m_queueLength;
    }
  
  reward /= reward_scale;
  NS_LOG_DEBUG ("MyGetReward: " << reward);

  m_reward_sum += reward;

  return reward;
}

std::string
MyGymEnv2::GetExtraInfo ()
{
  NS_LOG_FUNCTION (this);
  std::string myInfo = "a=1";

  return myInfo;
}

void
MyGymEnv2::SetCw (Ptr<Node> node, uint32_t cwValue)
{
  NS_ASSERT (m_algorithm == RL);
  Ptr<NetDevice> dev = node->GetDevice (0);
  Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice> (dev);
  Ptr<WifiMac> wifi_mac = wifi_dev->GetMac ();
  Ptr<ODcfAdhocWifiMac> odcf_mac = DynamicCast<ODcfAdhocWifiMac> (wifi_mac);
  PointerValue ptr;
  odcf_mac->GetAttribute ("ODcf", ptr);
  Ptr<ODcf> odcf = ptr.Get<ODcf> ();

  odcf->SetCw (cwValue);
}

bool
MyGymEnv2::ExecuteActions (Ptr<OpenGymDataContainer> actions)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_DEBUG ("MyExecuteActions: " << actions);

  uint32_t numAgents = m_agents.GetN ();

  Ptr<OpenGymTupleContainer> tuple = DynamicCast<OpenGymTupleContainer> (actions);

  for (uint32_t i = 0; i < numAgents; i++)
    {
      Ptr<Node> node = m_agents.Get (i);

      Ptr<OpenGymDiscreteContainer> action = DynamicCast<OpenGymDiscreteContainer> (tuple->Get (i));
      const double base = 2;
      uint32_t exponent = action->GetValue ();
      uint32_t cwSize = std::pow (base, exponent) - 1;
      SetCw (node, cwSize);
    }

  return true;
}

void
MyGymEnv2::SrcTxDone (Ptr<MyGymEnv2> entity, Ptr<Node> node, uint32_t idx, const WifiMacHeader &hdr)
{
  Packet *packet = (Packet *) hdr.m_packet;

  if (packet)
    {
      Ptr<MyGymNodeState> state = entity->m_agent_state[idx];
      state->m_txPktNum++;

      state->m_delay_estimator->RecordRx (packet); // MAQ delay

      Time delay = state->m_delay_estimator->GetLastDelay ();

      const double ewmaWeight = 0.9;
      if (state->m_txPktNum == 1)
        state->m_delayEwma = delay.GetDouble ();
      else
        state->m_delayEwma =
            ewmaWeight * state->m_delayEwma + (1 - ewmaWeight) * delay.GetDouble ();

      state->m_delaySum += delay;
    }
}

void
MyGymEnv2::PrintResults (void)
{
  uint32_t numAgents = m_agents.GetN ();

  double delaySum = 0.0;
  uint64_t pktSum = 0;
  for (uint32_t i = 0; i < numAgents; i++)
    {
      std::cout << m_agent_state[i]->m_txPktNum << ", ";
      delaySum += m_agent_state[i]->m_delaySum.GetDouble ();
      pktSum += m_agent_state[i]->m_txPktNum;
    }
  std::cout << std::endl;

  std::cout << "average end-to-end latency: " << delaySum / pktSum / 1e6 << " ms" << std::endl; // milliseconds
  std::cout << "episode_reward: " << m_reward_sum * numAgents << std::endl;
}

} // namespace ns3