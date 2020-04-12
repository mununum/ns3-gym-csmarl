#include "mygym.h"
#include "ns3/object.h"
#include "ns3/core-module.h"
#include "ns3/wifi-module.h"
#include "ns3/node-list.h"
#include "ns3/log.h"
#include "ns3/delay-jitter-estimation.h"
#include <sstream>
#include <iostream>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("MyGymEnv");

NS_OBJECT_ENSURE_REGISTERED (MyGymEnv);

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

  friend class MyGymEnv;

public:
  MyGymNodeState ()
      : m_txPktNum (0),
        m_txPktNumLastVal (0),
        // m_rxPktNum (0),
        // m_rxPktNumLastVal (0),
        m_txPktNumMovingAverage (0.0),
        // m_rxPktNumMovingAverage (0.0),
        m_delaySum (Seconds (0.0)),
        m_delay_estimator (CreateObject<DelayJitterEstimation> ())
  {
  }

  void
  Reset ()
  {
    updateEwma (m_txPktNum, m_txPktNumLastVal, m_txPktNumMovingAverage, 0.9);
    m_txPktNumLastVal = m_txPktNum;
    // m_rxPktNumLastVal = m_rxPktNum;
    m_delaySum = Seconds (0.0);
  }

  double
  getAvgTxDelay ()
  {
    uint64_t currentTxPktNum = m_txPktNum - m_txPktNumLastVal;
    return (currentTxPktNum > 0) ? (m_delaySum / currentTxPktNum).GetDouble () : 0.0;
  }

private:
  uint64_t m_txPktNum;
  uint64_t m_txPktNumLastVal;

  // uint64_t m_rxPktNum;
  // uint64_t m_rxPktNumLastVal;

  double m_txPktNumMovingAverage;
  // double m_rxPktNumMovingAverage;

  Time m_delaySum;
  Ptr<DelayJitterEstimation> m_delay_estimator;
};

MyGymEnv::MyGymEnv ()
{
  NS_LOG_FUNCTION (this);
}

MyGymEnv::MyGymEnv (NodeContainer agents, Time simTime, Time stepTime, bool enabled = true,
                    bool continuous = false)
{
  NS_LOG_FUNCTION (this);
  m_agents = agents;
  m_interval = stepTime;
  m_enabled = enabled;
  m_simTime = simTime;

  m_continuous = continuous;

  // initialize per-agent internal state
  uint32_t numAgents = m_agents.GetN ();
  for (uint32_t i = 0; i < numAgents; i++)
    {
      m_agent_state.push_back (CreateObject<MyGymNodeState> ());
    }

  m_perAgentObsDim = 5;  // Throughput, AvgThpt, Latency, Loss%, CW
  // m_perAgentObsDim++;  // agent_index

  Simulator::Schedule (Seconds (0.0), &MyGymEnv::ScheduleNextStateRead, this);
}

void
MyGymEnv::ScheduleNextStateRead ()
{
  NS_LOG_FUNCTION (this);

  // notify to python-end only when enabled
  if (m_enabled)
    Notify ();
  else
    {
      GetGameOver ();
      GetObservation ();
      GetReward ();
      GetExtraInfo ();
    }

  StepState ();

  Simulator::Schedule (m_interval, &MyGymEnv::ScheduleNextStateRead, this);
}

void
MyGymEnv::StepState ()
{
  uint32_t numAgents = m_agents.GetN ();

  for (uint32_t i = 0; i < numAgents; i++)
    {
      Ptr<Node> node = m_agents.Get (i);
      Ptr<NetDevice> dev = node->GetDevice (0);
      Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice> (dev);
      Ptr<WifiRemoteStationManager> rman = wifi_dev->GetRemoteStationManager ();

      // reset every statistics
      m_agent_state[i]->Reset ();
      rman->GetAggInfo ().Reset ();
    }
}

MyGymEnv::~MyGymEnv ()
{
  NS_LOG_FUNCTION (this);
}

TypeId
MyGymEnv::GetTypeId (void)
{
  static TypeId tid = TypeId ("MyGymEnv")
                          .SetParent<OpenGymEnv> ()
                          .SetGroupName ("OpenGym")
                          .AddConstructor<MyGymEnv> ();
  return tid;
}

uint32_t
MyGymEnv::GetTotalPkt ()
{
  uint32_t numAgents = m_agents.GetN ();
  uint32_t rxPktSum = 0;
  for (uint32_t i = 0; i < numAgents; i++)
    {
      rxPktSum += m_agent_state[i]->m_txPktNum;
      // rxPktSum += m_agent_state[i]->m_rxPktNum;
      // std::cout << "link " << i << " sent " << m_agent_state[i]->m_rxPktNum << " packets in "
      //           << m_simTime << std::endl;
    }
  // std::cout << rxPktSum << std::endl;
  return rxPktSum;
}

void
MyGymEnv::DoDispose ()
{
  NS_LOG_FUNCTION (this);
}

Ptr<OpenGymSpace>
MyGymEnv::GetActionSpace ()
{
  NS_LOG_FUNCTION (this);
  uint32_t nodeNum = m_agents.GetN ();
  // float low = 1.0;
  // float high = 1024.0;
  if (m_continuous)
    {
      std::vector<uint32_t> shape = {
          nodeNum,
      };
      float low = -1e5;
      float high = +1e5;
      std::string dtype = TypeNameGet<float> ();
      Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace> (low, high, shape, dtype);
      NS_LOG_DEBUG ("GetActionSpace: " << space);
      return space;
    }
  else
    {
      int n_actions = 10;
      Ptr<OpenGymTupleSpace> space = CreateObject<OpenGymTupleSpace> ();

      for (uint32_t i = 0; i < nodeNum; i++)
        {
          space->Add (CreateObject<OpenGymDiscreteSpace> (n_actions));
        }

      // float low = 0.0;
      // float high = 9.0;
      // std::string dtype = TypeNameGet<int32_t> ();
      // Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace> (low, high, shape, dtype);
      NS_LOG_DEBUG ("GetActionSpace: " << space);
      return space;
    }
}

Ptr<OpenGymSpace>
MyGymEnv::GetObservationSpace ()
{
  NS_LOG_FUNCTION (this);
  uint32_t agentNum = m_agents.GetN ();
  m_obs_shape = {agentNum, m_perAgentObsDim};

  float low = 0.0;
  float high = 10000.0;

  // std::vector<uint32_t> shape = {nodeNum,};
  std::string dtype = TypeNameGet<float> ();
  // Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace> (low, high, shape, dtype);
  Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace> (low, high, m_obs_shape, dtype);
  NS_LOG_DEBUG ("GetObservationSpace: " << space);
  return space;
}

bool
MyGymEnv::GetGameOver ()
{
  NS_LOG_FUNCTION (this);
  bool isGameOver = false;
  NS_LOG_DEBUG ("MyGetGameOver: " << isGameOver);
  return isGameOver;
}

Ptr<WifiMacQueue>
MyGymEnv::GetQueue (Ptr<Node> node)
{
  Ptr<NetDevice> dev = node->GetDevice (0);
  Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice> (dev);
  Ptr<WifiMac> wifi_mac = wifi_dev->GetMac ();
  Ptr<RegularWifiMac> rmac = DynamicCast<RegularWifiMac> (wifi_mac);
  PointerValue ptr;
  rmac->GetAttribute ("Txop", ptr);
  Ptr<Txop> txop = ptr.Get<Txop> ();
  Ptr<WifiMacQueue> queue = txop->GetWifiMacQueue ();
  return queue;
}

Ptr<OpenGymDataContainer>
MyGymEnv::GetObservation ()
{
  NS_LOG_FUNCTION (this);
  Ptr<OpenGymBoxContainer<float>> box = CreateObject<OpenGymBoxContainer<float>> (m_obs_shape);

  Time delaySum = Seconds (0.0);
  uint32_t pktSum = 0;

  NS_LOG_DEBUG (Simulator::Now () << " MyGetObservation:");

  uint32_t numAgents = m_agents.GetN ();
  for (uint32_t i = 0; i < numAgents; i++)
    {
      Ptr<Node> node = m_agents.Get (i);

      // NS_LOG_DEBUG ("# of pkts sent: " << m_agent_state[i]->m_txPktNum);

      // collect statistics
      // Throughput
      double thpt = m_agent_state[i]->m_txPktNum - m_agent_state[i]->m_txPktNumLastVal;
      thpt /= 1000;

      // XXX moving average?
      // double avg_thpt =
      //     updateEwma (m_agent_state[i]->m_txPktNum, m_agent_state[i]->m_txPktNumLastVal,
      //                 m_agent_state[i]->m_txPktNumMovingAverage, 0.9);
      double avg_thpt = m_agent_state[i]->m_txPktNumMovingAverage;
      avg_thpt /= 1000;

      // Latency
      double lat = m_agent_state[i]->getAvgTxDelay ();
      lat /= 1e9; // latency unit: sec

      delaySum += m_agent_state[i]->m_delaySum;
      pktSum += m_agent_state[i]->m_txPktNum;

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
      // double maxcw = txop->GetMaxCw ();

      // put it in a box
      box->AddValue (thpt);
      box->AddValue (avg_thpt);
      box->AddValue (lat);
      box->AddValue (err_rate);
      box->AddValue (mincw);

      NS_LOG_DEBUG (thpt << ",\t" << avg_thpt << ",\t" << lat << ",\t" << err_rate << ",\t" << mincw);

      // agent_id
      // box->AddValue (i);
    }

  // for (NodeList::Iterator i = NodeList::Begin (); i != NodeList::End (); ++i) {
  //     Ptr<Node> node = *i;
  //     Ptr<WifiMacQueue> queue = GetQueue (node);
  //     uint32_t value = queue->GetNPackets();
  //     box->AddValue(value);
  // }

  // NS_LOG_DEBUG ("MyGetObservation: " << box);
  return box;
}

float
MyGymEnv::GetReward ()
{
  NS_LOG_FUNCTION (this);
  const double epsilon = 5e-5;
  const double lower_bound = 0.1;
  const double reward_scale = 1000.0;

  float rate_reward = 0.0, rate_reward_min = 0.0;
  float delay_reward = 0.0;
  float loss_reward = 0.0;
  float reward = 0.0;

  for (uint32_t i = 0; i < m_agents.GetN (); i++)
    {
      {
        // double avg_rate =
        //     updateEwma (m_agent_state[i]->m_rxPktNum, m_agent_state[i]->m_rxPktNumLastVal,
        //                 m_agent_state[i]->m_rxPktNumMovingAverage, 0.9);
        double avg_rate = m_agent_state[i]->m_txPktNumMovingAverage;

        // log(5e-5) ~= -10, this prevents logarithm from being minus infinity
        rate_reward += std::log (avg_rate + epsilon);

        // the moving average will be always bigger than 0.1
        // rate_reward += std::log ( std::max (avg_rate, lower_bound) );

        rate_reward_min += std::log (lower_bound);

        // rate_reward += avg_rate / 1000.0;  // sum-rate reward function
      }
      {
        double avg_lat = m_agent_state[i]->getAvgTxDelay ();
        delay_reward += avg_lat / 1e9;  // seconds
      }
      {
        // MYTODO should make this into function?
        Ptr<Node> node = m_agents.Get (i);
        Ptr<NetDevice> dev = node->GetDevice (0);
        Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice> (dev);
        Ptr<WifiRemoteStationManager> rman = wifi_dev->GetRemoteStationManager ();
        double err_rate = rman->GetAggInfo ().GetFrameErrorRate ();
        loss_reward += err_rate;
      }
    }
  rate_reward = std::max (rate_reward, rate_reward_min);

  // reward = rate_reward - 100 * delay_reward - 200 * loss_reward;
  // reward = rate_reward - 100 * delay_reward;
  reward = rate_reward;
  m_rate_reward = rate_reward;
  m_delay_reward = delay_reward;

  NS_LOG_DEBUG ("rate_reward: " << rate_reward);
  NS_LOG_DEBUG ("delay_reward: " << delay_reward);
  // MYTODO export this to ExtraInfo
  NS_LOG_DEBUG ("loss_reward: " << loss_reward);

  reward /= reward_scale;
  NS_LOG_DEBUG ("MyGetReward: " << reward);
  return reward;
}

std::string
MyGymEnv::GetExtraInfo ()
{
  NS_LOG_FUNCTION (this);
  std::string myInfo = "";

  // for (uint32_t i = 0; i < m_agents.GetN (); i++)
  //   {
  //     myInfo += std::to_string (i);
  //     myInfo += "=";
  //     myInfo += std::to_string (
  //         (float) (m_agent_state[i]->m_rxPktNum - m_agent_state[i]->m_rxPktNumLastVal) / 1000.0);
  //     myInfo += " ";
  //   }
  
  myInfo += "rate_reward=" + std::to_string (m_rate_reward) + " ";
  myInfo += "delay_reward=" + std::to_string (m_delay_reward);

  NS_LOG_DEBUG ("MyGetExtraInfo: " << myInfo);
  return myInfo;
}

bool
MyGymEnv::SetCw (Ptr<Node> node, uint32_t cwMinValue, uint32_t cwMaxValue)
{
  Ptr<NetDevice> dev = node->GetDevice (0);
  Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice> (dev);
  Ptr<WifiMac> wifi_mac = wifi_dev->GetMac ();
  Ptr<RegularWifiMac> rmac = DynamicCast<RegularWifiMac> (wifi_mac);
  PointerValue ptr;
  rmac->GetAttribute ("Txop", ptr);
  Ptr<Txop> txop = ptr.Get<Txop> ();

  // if both set to the same value then we have uniform backoff?
  uint32_t minValue = 1.0;
  uint32_t maxValue = 1023.0;

  cwMinValue = std::max (cwMinValue, minValue);
  cwMinValue = std::min (cwMinValue, maxValue);
  txop->SetMinCw (cwMinValue);

  cwMaxValue = std::max (cwMaxValue, minValue);
  cwMaxValue = std::min (cwMaxValue, maxValue);
  txop->SetMaxCw (cwMaxValue);

  // if (cwMinValue > 0 && cwMinValue < 1024) {
  //     NS_LOG_DEBUG ("Set CW min: " << cwMinValue);
  //     txop->SetMinCw(cwMinValue);
  // }

  // if (cwMaxValue > 0 && cwMaxValue < 1024) {
  //     NS_LOG_DEBUG ("Set CW max: " << cwMaxValue);
  //     txop->SetMaxCw(cwMaxValue);
  // }

  return true;
}

bool
MyGymEnv::ExecuteActions (Ptr<OpenGymDataContainer> action)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_DEBUG ("MyExecuteActions: " << action);

  uint32_t agentNum = m_agents.GetN ();

  if (m_continuous)
    {

      Ptr<OpenGymBoxContainer<float>> box = DynamicCast<OpenGymBoxContainer<float>> (action);
      std::vector<float> actionVector = box->GetData ();

      for (uint32_t i = 0; i < agentNum; i++)
        {
          // Continuous action to change CW
          Ptr<Node> node = m_agents.Get (i);
          Ptr<NetDevice> dev = node->GetDevice (0);
          Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice> (dev);
          Ptr<WifiMac> wifi_mac = wifi_dev->GetMac ();
          Ptr<RegularWifiMac> rmac = DynamicCast<RegularWifiMac> (wifi_mac);
          PointerValue ptr;
          rmac->GetAttribute ("Txop", ptr);
          Ptr<Txop> txop = ptr.Get<Txop> ();
          double mincw = txop->GetMinCw ();

          float change_ratio = actionVector.at (i);
          float damping_factor = 0.01;
          if (change_ratio >= 0)
            {
              uint32_t newcw = mincw * (1 + change_ratio * damping_factor);
              SetCw (node, newcw, newcw);
            }
          else
            {
              uint32_t newcw = mincw / (1 - change_ratio * damping_factor);
              SetCw (node, newcw, newcw);
            }
        }
    }
  else
    {

      Ptr<OpenGymTupleContainer> tuple = DynamicCast<OpenGymTupleContainer> (action);

      for (uint32_t i = 0; i < agentNum; i++)
        {
          Ptr<Node> node = m_agents.Get (i);

          Ptr<OpenGymDiscreteContainer> action_i =
              DynamicCast<OpenGymDiscreteContainer> (tuple->Get (i));
          uint32_t exponent = action_i->GetValue ();
          uint32_t cwSize = std::pow (2, exponent) - 1;
          SetCw (node, cwSize, cwSize);
        }
    }

  return true;
}

// for event-based env
// void
// MyGymEnv::NotifyPktRxEvent(Ptr<MyGymEnv> entity, Ptr<Node> node, Ptr<const Packet> packet)
// {
//     NS_LOG_DEBUG ("Client received a packet of " << packet->GetSize () << " bytes");
//     entity->m_currentNode = node;
//     entity->m_rxPktNum++;

//     NS_LOG_UNCOND ("Node with ID " << entity->m_currentNode->GetId() << " received " << entity->m_rxPktNum << " packets");

//     entity->Notify();
// }

// This method is called upon RX
void
MyGymEnv::CountRxPkts (Ptr<MyGymEnv> entity, Ptr<Node> node, uint32_t idx, Ptr<const Packet> packet)
{
  // NS_LOG_DEBUG ("Client received a packet of " << packet->GetSize () << " bytes");
  // entity->m_currentNode = node;
  // entity->m_agent_state[idx]->m_rxPktNum++;
  // entity->m_rxPktNum++;

  // entity->m_delay_estimator->RecordRx (packet);

  // NS_LOG_DEBUG ("Node with ID " << node->GetId()
  //             //    << " received " << entity->m_rxPktNum << " packets, "
  //                << " received packet " << packet->GetUid() << " with size " << packet->GetSize() << ", "
  //                << "estimated delay: " << entity->m_delay_estimator->GetLastDelay ());
}

// This method is called upon ACK receiving
void
MyGymEnv::SrcTxDone (Ptr<MyGymEnv> entity, Ptr<Node> node, uint32_t idx, const WifiMacHeader &hdr)
{
  Packet *packet = (Packet *) hdr.m_packet;

  if (packet)
    {
      // record TX throughput and latency only when it is a unicast frame
      Ptr<MyGymNodeState> state = entity->m_agent_state[idx];
      state->m_txPktNum++;
      // state->m_txPktBytes += packet->GetSize ();
      state->m_delay_estimator->RecordRx (packet);
      state->m_delaySum += state->m_delay_estimator->GetLastDelay ();
    }

  // entity->m_delay_estimator->RecordRx (packet);

  // NS_LOG_UNCOND ("Node with ID " << node->GetId()
  //                << " sent packet " << packet->GetUid() << " with size " << packet->GetSize() << ", "
  //                << "estimated delay: " << entity->m_delay_estimator->GetLastDelay () << ", "
  //                << "error rate: " << error_rate);
}

// This method is called upon Tx fail
void
MyGymEnv::SrcTxFail (Ptr<MyGymEnv> entity, Ptr<Node> node, uint32_t idx, const WifiMacHeader &hdr)
{
  Packet *packet = (Packet *) hdr.m_packet;

  if (packet)
    {
      NS_LOG_DEBUG ("Node with ID " << node->GetId () << " has failed to send a packet with size "
                                    << packet->GetSize ());
    }
}

} // namespace ns3
