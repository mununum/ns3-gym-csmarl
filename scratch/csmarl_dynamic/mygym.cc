#include "ns3/core-module.h"
#include "ns3/wifi-module.h"
#include "ns3/delay-jitter-estimation-2.h"
#include "ns3/udp-client.h"
#include "mygym.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("MyGymEnv");

NS_OBJECT_ENSURE_REGISTERED (MyGymEnv);

static inline double
updateEwma (const uint64_t value, const uint64_t lastValue, double ewmaValue, const double ewmaWeight)
{
  double currentValue = value - lastValue;
  if (lastValue == 0)
    return currentValue;
  else
    return ewmaWeight * ewmaValue + (1 - ewmaWeight) * currentValue;
}

class MyGymAgent : public Object
{
  friend class MyGymEnv;

public:
  MyGymAgent (Ptr<Node> node)
      : m_node (node),
        m_txPktNum (0),
        m_txPktNumLastVal (0),
        m_txPktNumEwma (0.0),
        m_queueLength (0),
        m_e2eDelayEwma (0.0),
        m_totalCCADuration (Seconds (0.0)),
        m_busyCCADuration (Seconds (0.0)),
        m_delayEstimator (CreateObject<DelayJitterEstimation2> (1))  // delay from MAQ enqueue
  {
  }

  void
  Step ()
  {
    m_txPktNumEwma = updateEwma (m_txPktNum, m_txPktNumLastVal, m_txPktNumEwma, 0.9);
    m_txPktNumLastVal = m_txPktNum;

    m_totalCCADuration = Seconds (0.0);
    m_busyCCADuration = Seconds (0.0);
  }

protected:
  Ptr<Node> m_node;

  uint64_t m_txPktNum;
  uint64_t m_txPktNumLastVal;

  double m_txPktNumEwma;

  uint32_t m_queueLength;  // MYTODO is this used?

  double m_e2eDelayEwma;

  Time m_totalCCADuration;
  Time m_busyCCADuration;

  Ptr<DelayJitterEstimation2> m_delayEstimator;  // for HOL delay
};

MyGymEnv::MyGymEnv ()
{
  NS_LOG_FUNCTION (this);
}

MyGymEnv::MyGymEnv (NodeContainer &srcNodes, Time stepTime, std::string algorithm, bool debug)
{
  NS_LOG_FUNCTION (this);

  m_stepTime = stepTime;

  if (algorithm == "80211")
    m_algorithm = IEEE80211;
  else if (algorithm == "odcf")
    m_algorithm = O_DCF;
  else if (algorithm == "rl")
    m_algorithm = RL;
  else
    NS_FATAL_ERROR ("invalid algorithm");

  m_debug = debug;

  for (auto it = srcNodes.Begin (); it != srcNodes.End (); it++)
    m_agents.push_back (CreateObject<MyGymAgent> (*it));

  m_numAgents = srcNodes.GetN ();
  m_obsShape = {7};  // thpt, qlen, lat, err_rate, busy_cca, cw, interval

  m_utilityReward = 0.0;
  m_queueReward = 0.0;

  m_utilityRewardSum = 0.0;
  m_queueRewardSum = 0.0;

  Simulator::Schedule (Seconds (0.0), &MyGymEnv::ScheduleNextStateRead, this);
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

void
MyGymEnv::DoDispose ()
{
  NS_LOG_FUNCTION (this);
}

void
MyGymEnv::ScheduleNextStateRead ()
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

  Simulator::Schedule (m_stepTime, &MyGymEnv::ScheduleNextStateRead, this);
}

void
MyGymEnv::StepState ()
{
  for (auto agent : m_agents)
    {
      Ptr<NetDevice> dev = agent->m_node->GetDevice (0);
      Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice> (dev);
      Ptr<WifiRemoteStationManager> rman = wifi_dev->GetRemoteStationManager ();

      // reset every statistics
      agent->Step ();
      rman->GetAggInfo ().Reset ();
    }
}

Ptr<OpenGymSpace>
MyGymEnv::GetActionSpace ()
{
  NS_LOG_FUNCTION (this);

  const uint32_t n_actions = 9;
  Ptr<OpenGymTupleSpace> space = CreateObject<OpenGymTupleSpace> ();

  for (uint32_t i = 0; i < m_numAgents; i++)
    {
      Ptr<OpenGymTupleSpace> perAgentTuple = CreateObject<OpenGymTupleSpace> ();
      perAgentTuple->Add (CreateObject<OpenGymDiscreteSpace> (n_actions));  // for cw control
      perAgentTuple->Add (CreateObject<OpenGymDiscreteSpace> (10));         // for rate control
      space->Add (perAgentTuple);
    }

  NS_LOG_DEBUG ("GetActionSpace: " << space);
  return space;
}

Ptr<OpenGymSpace>
MyGymEnv::GetObservationSpace ()
{
  NS_LOG_FUNCTION (this);

  float low = 0;
  float high = 10000.0;

  std::string dtype = TypeNameGet<float> ();
  Ptr<OpenGymTupleSpace> space = CreateObject<OpenGymTupleSpace> ();

  for (uint32_t i = 0; i < m_numAgents; i++)
    {
      Ptr<OpenGymBoxSpace> box = CreateObject<OpenGymBoxSpace> (low, high, m_obsShape, dtype);
      space->Add (box);
    }

  NS_LOG_DEBUG ("GetObservationSpace: " << space);
  return space;
}

bool
MyGymEnv::GetGameOver ()
{
  NS_LOG_FUNCTION (this);
  bool isGameOver = false;
  NS_LOG_DEBUG ("GetGameOver: " << isGameOver);
  return isGameOver;
}

static Ptr<ODcf>
GetODcf (Ptr<Node> node)
{
  Ptr<NetDevice> dev = node->GetDevice (0);
  Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice> (dev);
  Ptr<WifiMac> wifi_mac = wifi_dev->GetMac ();
  Ptr<ODcfAdhocWifiMac> odcf_mac = DynamicCast<ODcfAdhocWifiMac> (wifi_mac);
  PointerValue ptr;
  odcf_mac->GetAttribute ("ODcf", ptr);
  Ptr<ODcf> odcf = ptr.Get<ODcf> ();
  return odcf;
}

static Ptr<Txop>
GetTxop (Ptr<Node> node)
{
  Ptr<NetDevice> dev = node->GetDevice (0);
  Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice> (dev);
  Ptr<WifiMac> wifi_mac = wifi_dev->GetMac (); 
  Ptr<RegularWifiMac> rmac = DynamicCast<RegularWifiMac> (wifi_mac);
  PointerValue ptr;
  rmac->GetAttribute ("Txop", ptr);
  Ptr<Txop> txop = ptr.Get<Txop> ();
  return txop;
}

uint32_t
MyGymEnv::GetQueueLength (Ptr<Node> node)
{
  if (m_algorithm == IEEE80211)
    {
      // 80211: Get IFQ length
      return GetTxop (node)->GetWifiMacQueue ()->GetNPackets ();
    }
  else
    {
      // O-DCF/RL: Get MAQ length
      return GetODcf (node)->GetMAQLength ();
    }
}

Ptr<OpenGymDataContainer>
MyGymEnv::GetObservation ()
{
  NS_LOG_FUNCTION (this);

  Ptr<OpenGymTupleContainer> tuple = CreateObject<OpenGymTupleContainer> ();

  for (auto agent : m_agents)
    {
      Ptr<OpenGymBoxContainer<float>> box = CreateObject<OpenGymBoxContainer<float>> (m_obsShape);
      Ptr<Node> node = agent->m_node;

      // Throughput
      double thpt = agent->m_txPktNum - agent->m_txPktNumLastVal;
      thpt /= 1000;

      // Queue length
      uint32_t qlen = GetQueueLength (node);

      // Average throughput

      // HOL latency
      double lat = agent->m_e2eDelayEwma;
      lat /= 1e9;

      // Busy CCA duration
      double busy_cca_fraction;
      if (agent->m_totalCCADuration == Seconds (0.0))
        busy_cca_fraction = 0.0;
      else
        busy_cca_fraction = agent->m_busyCCADuration.GetDouble () / agent->m_totalCCADuration.GetDouble ();

      // Loss rate
      Ptr<NetDevice> dev = node->GetDevice (0);
      Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice> (dev);
      Ptr<WifiRemoteStationManager> rman = wifi_dev->GetRemoteStationManager ();
      double err_rate = rman->GetAggInfo ().GetFrameErrorRate ();

      // CW
      double mincw = GetCw (node);

      // sourceInterval
      double interval = GetSourceInterval (node).GetDouble () / 1e6;

      box->AddValue (thpt);
      box->AddValue (qlen);
      box->AddValue (lat);
      box->AddValue (err_rate);
      box->AddValue (busy_cca_fraction);
      box->AddValue (mincw);
      box->AddValue (interval);

      tuple->Add (box);

      if (m_debug)
        std::cout << qlen << "\t";
      // if (m_debug)
      //   std::cout << agent->m_txPktNumEwma << "\t";
    }

  if (m_debug)
    std::cout << std::endl;

  return tuple;
}

float
MyGymEnv::GetReward ()
{
  NS_LOG_FUNCTION (this);

  const double epsilon = 1e-5;
  float utility_reward = 0.0;
  float queue_reward = 0.0;

  for (auto agent : m_agents)
    {
      utility_reward += std::log (agent->m_txPktNumEwma + epsilon);
      // utility_reward *= agent->m_txPktNumEwma;
      queue_reward -= GetQueueLength (agent->m_node);
    }

  queue_reward /= m_numAgents;

  utility_reward /= 1e3;
  queue_reward /= 1e5;

  m_utilityReward = utility_reward;
  m_queueReward = queue_reward;

  m_utilityRewardSum += utility_reward;
  m_queueRewardSum += queue_reward;

  // if (m_debug)
  //   std::cout << utility_reward << std::endl;
  
  return utility_reward + queue_reward;
}

std::string
MyGymEnv::GetExtraInfo ()
{
  NS_LOG_FUNCTION (this);
  std::string myInfo = "";

  myInfo += "utility=" + std::to_string (m_utilityReward) + " ";
  myInfo += "queue=" + std::to_string (m_queueReward);

  return myInfo;
}

void
MyGymEnv::SetCw (Ptr<Node> node, uint32_t cwValue)
{
  NS_ASSERT (m_algorithm == RL);

  GetODcf (node)->SetCw (cwValue);
}

uint32_t
MyGymEnv::GetCw (Ptr<Node> node)
{
  if (m_algorithm == IEEE80211)
    {
      // 80211: refer to Txop object
      return GetTxop (node)->GetMinCw ();
    }
  else
    {
      // ODCF/RL: refer to ODcf object
      return GetODcf (node)->GetCw ();
    }
}

void
MyGymEnv::SetSourceInterval (Ptr<Node> node, Time interval)
{
  NS_ASSERT (m_algorithm == RL);

  GetODcf (node)->SetSourceInterval (interval);
}

Time
MyGymEnv::GetSourceInterval (Ptr<Node> node)
{
  if (m_algorithm == IEEE80211)
    {
      Ptr<Application> app = node->GetApplication (0);
      Ptr<UdpClient> udp = DynamicCast<UdpClient> (app);
      TimeValue time;
      udp->GetAttribute ("Interval", time);
      return time.Get ();
    }
  else
    {
      return GetODcf (node)->GetSourceInterval (0);
    }
}

bool
MyGymEnv::ExecuteActions (Ptr<OpenGymDataContainer> actions)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_DEBUG ("ExecuteActions: " << actions);

  Ptr<OpenGymTupleContainer> tuple = DynamicCast<OpenGymTupleContainer> (actions);

  for (uint32_t i = 0; i < m_numAgents; i++)
    {
      Ptr<OpenGymTupleContainer> action = DynamicCast<OpenGymTupleContainer> (tuple->Get (i));

      Ptr<OpenGymDiscreteContainer> cwAction = DynamicCast<OpenGymDiscreteContainer> (action->Get (0));
      Ptr<OpenGymDiscreteContainer> rateAction = DynamicCast<OpenGymDiscreteContainer> (action->Get (1));

      // set mac parameter
      const double base = 2.0;
      uint32_t exponent = cwAction->GetValue ();
      uint32_t cwSize = std::pow (base, exponent) - 1;
      SetCw (m_agents[i]->m_node, cwSize);

      // set rate parameter
      uint32_t rate = rateAction->GetValue ();
      double pps = 10.0 + rate * 100.0;  // 10 ~ 910
      Time interval = Seconds (1.0 / pps);
      SetSourceInterval (m_agents[i]->m_node, interval);
    }

  return true;
}

void
MyGymEnv::SrcTxDone (Ptr<MyGymEnv> entity, uint32_t idx, const WifiMacHeader &hdr)
{
  Packet *packet = (Packet *) hdr.m_packet;

  if (packet)
    {
      Ptr<MyGymAgent> agent = entity->m_agents[idx];
      agent->m_txPktNum++;

      agent->m_delayEstimator->RecordRx (packet);  // HOL delay
      Time e2eDelay = agent->m_delayEstimator->GetLastDelay ();

      // EWMA update the HOL delay
      const double ewmaWeight = 0.9;
      if (agent->m_txPktNum == 1)
        agent->m_e2eDelayEwma = e2eDelay.GetDouble ();
      else
        agent->m_e2eDelayEwma =
          ewmaWeight * agent->m_e2eDelayEwma + (1 - ewmaWeight) * e2eDelay.GetDouble ();
    }
}

void
MyGymEnv::PhyStateChange (Ptr<MyGymEnv> entity, uint32_t idx, Time start, Time duration,
                          WifiPhyState state)
{
  Ptr<MyGymAgent> agent = entity->m_agents[idx];

  if (state == WifiPhyState::RX || state == WifiPhyState::CCA_BUSY)
    agent->m_busyCCADuration += duration;
  agent->m_totalCCADuration += duration;
}

void
MyGymEnv::PrintResults ()
{
  for (auto agent : m_agents)
    {
      std::cout << agent->m_txPktNum << "\t";
    }
  std::cout << std::endl;
  std::cout << "utility reward: " << m_utilityRewardSum << std::endl;
  std::cout << "queue reward: " << m_queueRewardSum << std::endl;
  std::cout << "episode reward: " << (m_utilityRewardSum + m_queueRewardSum) * m_numAgents << std::endl;
}

} // namespace ns3