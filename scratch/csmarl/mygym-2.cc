#include "mygym-2.h"
#include "ns3/object.h"
#include "ns3/core-module.h"
#include "ns3/wifi-module.h"
#include "ns3/node-list.h"
#include "ns3/log.h"
#include "ns3/delay-jitter-estimation.h"
#include "ns3/delay-jitter-estimation-2.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("MyGymEnv2");

NS_OBJECT_ENSURE_REGISTERED (MyGymEnv2);

static inline double
updateEwma (uint64_t &value, uint64_t &lastValue, double &ewmaValue, const double ewmaWeight)
{
  double currentValue = value - lastValue;
  if (lastValue == 0)
    {
      ewmaValue = currentValue;
    }
  else
    {
      ewmaValue = ewmaWeight * ewmaValue + (1 - ewmaWeight) * currentValue;
    }
  return ewmaValue;
}

class MyGymNodeState : public Object
{
  friend class MyGymEnv2;

public:
  MyGymNodeState ()
      : m_txPktNum (0),
        m_txPktNumLastVal (0),
        m_txPktNumEwma (0.0),
        m_tempReward (0.0),
        m_currentReward (0.0),
        m_queueLength (0),
        m_e2eDelaySum (Seconds (0.0)),
        m_e2eDelayEwma (0.0),
        m_HOLDelayEwma (0.0),
        m_totalCCADuration (Seconds (0.0)),
        m_busyCCADuration (Seconds (0.0)),
        m_delay_estimator (CreateObject<DelayJitterEstimation> ()),
        m_delay_estimator_2 (CreateObject<DelayJitterEstimation2> (1))
  {
  }

  void
  Step ()
  {
    updateEwma (m_txPktNum, m_txPktNumLastVal, m_txPktNumEwma, 0.9);
    m_txPktNumLastVal = m_txPktNum;

    m_tempReward = 0.0;
    m_currentReward = 0.0;

    m_totalCCADuration = Seconds (0.0);
    m_busyCCADuration = Seconds (0.0);
  }

private:
  uint64_t m_txPktNum;
  uint64_t m_txPktNumLastVal;

  double m_txPktNumEwma;

  float m_tempReward;
  float m_currentReward;

  uint32_t m_queueLength;

  Time m_e2eDelaySum;
  double m_e2eDelayEwma;
  double m_HOLDelayEwma;

  Time m_totalCCADuration;
  Time m_busyCCADuration;

  Ptr<DelayJitterEstimation> m_delay_estimator;  // for HOL delay
  Ptr<DelayJitterEstimation2> m_delay_estimator_2; // for MAQ delay
};

MyGymEnv2::MyGymEnv2 ()
{
  NS_LOG_FUNCTION (this);
}

MyGymEnv2::MyGymEnv2 (NodeContainer agents, Time stepTime, std::string algorithm,
                      std::map<uint32_t, std::set<uint32_t>> neighbors,
                      std::map<uint32_t, uint32_t> degree,
                      std::map<uint32_t, double> neiInvDegSum,
                      bool debug = false)
{
  NS_LOG_FUNCTION (this);
  m_agents = agents;
  m_stepTime = stepTime;

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
  m_reward_indiv_sum = 0.0;

  uint32_t numAgents = m_agents.GetN ();
  for (uint32_t i = 0; i < numAgents; i++)
    {
      m_agent_state.push_back (CreateObject<MyGymNodeState> ());
    }

  m_perAgentObsDim = 6; // Throughput, QueueLength, e2e Latency, Loss%, BusyCCAFrac, CW

  m_neighbors = neighbors;
  m_degree = degree;
  m_neiInvDegSum = neiInvDegSum;

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

  Simulator::Schedule (m_stepTime, &MyGymEnv2::ScheduleNextStateRead, this);
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

      // AvgThpt
      double avg_thpt = m_agent_state[i]->m_txPktNumEwma;
      avg_thpt /= 1000;

      // e2e latency
      // double lat = m_agent_state[i]->m_e2eDelayEwma;
      // lat /= 1e9; // latency unit: sec

      // HOL latency
      double lat = m_agent_state[i]->m_HOLDelayEwma;
      lat /= 1e9; // latency unit: sec

      // busy cca duration
      double busy_cca_fraction;
      if (m_agent_state[i]->m_totalCCADuration == Seconds (0.0))
        busy_cca_fraction = 0.0;
      else
        busy_cca_fraction = m_agent_state[i]->m_busyCCADuration.GetDouble () / m_agent_state[i]->m_totalCCADuration.GetDouble ();

      // Loss%
      Ptr<NetDevice> dev = node->GetDevice (0);
      Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice> (dev);
      Ptr<WifiRemoteStationManager> rman = wifi_dev->GetRemoteStationManager ();
      double err_rate = rman->GetAggInfo ().GetFrameErrorRate ();

      // CW
      double mincw = GetCw (node);

      // put it in a box
      box->AddValue (thpt);
      // box->AddValue (qlen);
      box->AddValue (avg_thpt);
      box->AddValue (lat);
      box->AddValue (err_rate);
      box->AddValue (busy_cca_fraction);
      box->AddValue (mincw);

      if (m_debug)
        std::cout << qlen << "\t";
        // std::cout << busy_cca_fraction << "\t";
        // std::cout << thpt << "\t" << qlen << "\t" << lat << "\t" << err_rate << "\t" << mincw << std::endl;
    }

  if (m_debug)
    std::cout << std::endl;

  return box;
}

float
MyGymEnv2::GetReward ()
{
  NS_LOG_FUNCTION (this);

  const double queue_reward_scale = 1e2;
  const double utility_reward_scale = 30;
  const double reward_scale = 1e3;
  const double epsilon = 5e-5;
  const double lower_bound = 0.1;

  uint32_t numAgents = m_agents.GetN ();
  float reward = 0.0;
  float queue_reward = 0.0;
  float utility_reward = 0.0, utility_reward_min = 0.0;

  // shared reward
  utility_reward_min = std::log (lower_bound);
  for (uint32_t i = 0; i < numAgents; i++)
    {
      double avg_rate = m_agent_state[i]->m_txPktNumEwma;
      utility_reward += std::log (avg_rate + epsilon);
      // utility_reward_min += std::log (lower_bound);

      m_agent_state[i]->m_tempReward = std::log (avg_rate + epsilon);

      queue_reward -= m_agent_state[i]->m_queueLength;
    }
  utility_reward = std::max (utility_reward, utility_reward_min);

  // individual reward (from graph)
  for (uint32_t i = 0; i < numAgents; i++)
    {
      m_agent_state[i]->m_currentReward = m_agent_state[i]->m_tempReward;
      // for (auto it = m_neighbors[i].begin (); it != m_neighbors[i].end (); it++) {
      //   m_agent_state[i]->m_currentReward += m_agent_state[*it]->m_tempReward / m_degree[*it];
      // }
      // m_agent_state[i]->m_currentReward /= 1 + m_neiInvDegSum[i];
      m_agent_state[i]->m_currentReward = std::max (m_agent_state[i]->m_currentReward, utility_reward_min);
      m_agent_state[i]->m_currentReward *= utility_reward_scale / reward_scale;
      m_reward_indiv_sum += m_agent_state[i]->m_currentReward;
    }

  // print individual reward coefficients
  // for (uint32_t i = 0; i < numAgents; i++)
  //   {
  //     std::cout << "R_" << i << " = (";
  //     for (auto it = m_neighbors[i].begin (); it != m_neighbors[i].end (); it++) {
  //       std::cout << "r_" << *it << " / " << m_degree[*it] << " + ";
  //     }
  //     std::cout << "r_" << i << ") ";
  //     std::cout << "/ " << (1 + m_neiInvDegSum[i]) << std::endl;
  //   }
  // exit(0);

  m_queue_reward = queue_reward / queue_reward_scale / reward_scale;
  m_utility_reward = utility_reward * utility_reward_scale / reward_scale;

  // if (m_useShortReward)
  //   reward = m_utility_reward;  // short_term_utility
  // else
  //   reward = m_queue_reward;
  reward = m_utility_reward;

  NS_LOG_DEBUG ("MyGetReward: " << reward);

  m_reward_sum += reward;

  return reward;
}

std::string
MyGymEnv2::GetExtraInfo ()
{
  NS_LOG_FUNCTION (this);
  std::string myInfo = "";

  // report individual rewards
  uint32_t numAgents = m_agents.GetN ();
  for (uint32_t i = 0; i < numAgents; i++)
    {
      myInfo += "reward_" + std::to_string(i) + "=" + std::to_string(m_agent_state[i]->m_currentReward) + " ";
    }

  myInfo += "queue_reward=" + std::to_string (m_queue_reward) + " ";
  myInfo += "utility_reward=" + std::to_string (m_utility_reward);

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

uint32_t
MyGymEnv2::GetCw (Ptr<Node> node)
{
  Ptr<NetDevice> dev = node->GetDevice (0);
  Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice> (dev);
  Ptr<WifiMac> wifi_mac = wifi_dev->GetMac ();
  if (m_algorithm == IEEE80211)
    {
      // 80211: refer to Txop object
      Ptr<RegularWifiMac> rmac = DynamicCast<RegularWifiMac> (wifi_mac);
      PointerValue ptr;
      rmac->GetAttribute ("Txop", ptr);
      Ptr<Txop> txop = ptr.Get<Txop> ();
      return txop->GetMinCw ();
    }
  else
    {
      // ODCF/RL: refer to ODcf object
      Ptr<ODcfAdhocWifiMac> odcf_mac = DynamicCast<ODcfAdhocWifiMac> (wifi_mac);
      PointerValue ptr;
      odcf_mac->GetAttribute ("ODcf", ptr);
      Ptr<ODcf> odcf = ptr.Get<ODcf> ();
      return odcf->GetCw ();
    }
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

      // if (m_debug)
      //   std::cout << exponent << "\t";
    }
  // if (m_debug)
  //   std::cout << std::endl;

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

      state->m_delay_estimator->RecordRx (packet);  // HOL delay
      state->m_delay_estimator_2->RecordRx (packet); // MAQ delay

      Time HOLDelay = state->m_delay_estimator->GetLastDelay ();
      Time e2eDelay = state->m_delay_estimator_2->GetLastDelay ();

      // EWMA update the HOL delay
      const double ewmaWeight = 0.9;
      if (state->m_txPktNum == 1)
        state->m_HOLDelayEwma = HOLDelay.GetDouble ();
      else
        state->m_HOLDelayEwma =
            ewmaWeight * state->m_HOLDelayEwma + (1 - ewmaWeight) * HOLDelay.GetDouble ();

      // EWMA update the e2e delay
      if (state->m_txPktNum == 1)
        state->m_e2eDelayEwma = e2eDelay.GetDouble ();
      else
        state->m_e2eDelayEwma =
            ewmaWeight * state->m_e2eDelayEwma + (1 - ewmaWeight) * e2eDelay.GetDouble ();

      // e2e delay statistic for performance measurement
      state->m_e2eDelaySum += e2eDelay;
    }
}

void
MyGymEnv2::PhyStateChange (Ptr<MyGymEnv2> entity, uint32_t idx, Time start, Time duration, WifiPhyState state)
{
  Ptr<MyGymNodeState> gym_state = entity->m_agent_state[idx];

  if (state == WifiPhyState::RX || state == WifiPhyState::CCA_BUSY)
    gym_state->m_busyCCADuration += duration;
  gym_state->m_totalCCADuration += duration;
}

void
MyGymEnv2::PrintResults (void)
{
  uint32_t numAgents = m_agents.GetN ();

  double delaySum = 0.0;
  uint64_t pktSum = 0;
  std::cout << "average throughput" << std::endl;
  for (uint32_t i = 0; i < numAgents; i++)
    {
      std::cout << m_agent_state[i]->m_txPktNum << "\t";
      delaySum += m_agent_state[i]->m_e2eDelaySum.GetDouble ();
      pktSum += m_agent_state[i]->m_txPktNum;
    }
  std::cout << std::endl;

  std::cout << "average end-to-end latency: " << delaySum / pktSum / 1e6 << " ms"
            << std::endl; // milliseconds
  std::cout << "episode_reward: " << m_reward_sum * numAgents << std::endl;
  std::cout << "episode_reward_indiv: " << m_reward_indiv_sum << std::endl;
}

} // namespace ns3