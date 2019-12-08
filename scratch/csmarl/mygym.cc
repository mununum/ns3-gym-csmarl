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

class MyGymNodeState : public Object {

    friend class MyGymEnv;

public:
    MyGymNodeState () : m_txPktBytes (0), m_txPktCount (0), m_delaySum (Seconds (0.0)),
                        m_delay_estimator (CreateObject<DelayJitterEstimation> ()) {}

    void Reset ()
    {
        m_txPktBytes = 0;
        m_txPktCount = 0;
        m_delaySum = Seconds (0.0);
    }

private:
    uint64_t m_txPktBytes;
    uint64_t m_txPktCount;
    Time m_delaySum;
    Ptr<DelayJitterEstimation> m_delay_estimator;

};

NS_LOG_COMPONENT_DEFINE ("MyGymEnv");

NS_OBJECT_ENSURE_REGISTERED (MyGymEnv);

// for event-based env
// MyGymEnv::MyGymEnv ()
// {
//     NS_LOG_FUNCTION (this);
//     m_currentNode = 0;
//     m_rxPktNum = 0;
// }
MyGymEnv::MyGymEnv ()
{
    NS_LOG_FUNCTION (this);
}

MyGymEnv::MyGymEnv (NodeContainer agents, Time stepTime, bool enabled=true)
{
    NS_LOG_FUNCTION (this);
    // m_currentNode = 0;
    m_rxPktNum = 0;
    m_agents = agents;
    m_interval = stepTime;
    m_enabled = enabled;
    
    // initialize per-agent internal state
    uint32_t numAgents = m_agents.GetN ();
    for (uint32_t i = 0; i < numAgents; i++) {
        m_agent_state.push_back (CreateObject<MyGymNodeState> ());
    }

    Simulator::Schedule (Seconds(0.0), &MyGymEnv::ScheduleNextStateRead, this);
}

void
MyGymEnv::ScheduleNextStateRead ()
{
    NS_LOG_FUNCTION (this);
    Simulator::Schedule (m_interval, &MyGymEnv::ScheduleNextStateRead, this);

    // notify to python-end only when enabled
    if (m_enabled)
        Notify();
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
        .AddConstructor<MyGymEnv> ()
    ;
    return tid;
}

void
MyGymEnv::DoDispose ()
{
    NS_LOG_FUNCTION (this);
}


Ptr<OpenGymSpace>
MyGymEnv::GetActionSpace()
{
    NS_LOG_FUNCTION (this);
    uint32_t nodeNum = m_agents.GetN ();
    float low = 0.0;
    float high = 100.0;
    std::vector<uint32_t> shape = {nodeNum,};
    std::string dtype = TypeNameGet<uint32_t> ();
    Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace> (low, high, shape, dtype);
    NS_LOG_UNCOND ("GetActionSpace: " << space);
    return space;
}

Ptr<OpenGymSpace>
MyGymEnv::GetObservationSpace()
{
    NS_LOG_FUNCTION (this);
    uint32_t nodeNum = m_agents.GetN ();
    uint32_t perNodeObsDim = 4; // Throughput, Latency, Loss%, CW
    m_obs_shape = {nodeNum, perNodeObsDim};

    // TODO change
    float low = 0.0;
    float high = 100.0;

    // std::vector<uint32_t> shape = {nodeNum,};
    std::string dtype = TypeNameGet<double> ();
    // Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace> (low, high, shape, dtype);
    Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace> (low, high, m_obs_shape, dtype);
    NS_LOG_UNCOND ("GetObservationSpace: " << space);
    return space;
}

bool
MyGymEnv::GetGameOver()
{
    NS_LOG_FUNCTION (this);
    bool isGameOver = false;
    NS_LOG_UNCOND ("MyGetGameOver: " << isGameOver);
    return isGameOver;
}

Ptr<WifiMacQueue>
MyGymEnv::GetQueue(Ptr<Node> node)
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
MyGymEnv::GetObservation()
{
    NS_LOG_FUNCTION (this);
    // uint32_t nodeNum = NodeList::GetNNodes ();
    Ptr<OpenGymBoxContainer<double>> box = CreateObject<OpenGymBoxContainer<double>>(m_obs_shape);

    uint32_t numAgents = m_agents.GetN ();
    for (uint32_t i = 0; i < numAgents; i++) {
        Ptr<Node> node = m_agents.Get (i);

        // collect statistics
        // TODO make proper normalization
        // Throughput
        double thpt = m_agent_state[i]->m_txPktBytes / 1000;
        // Latency
        double lat;
        if (m_agent_state[i]->m_txPktCount > 0) {
            lat = (m_agent_state[i]->m_delaySum / m_agent_state[i]->m_txPktCount).GetDouble ();
        } else {
            lat = 0;
        }
        lat /= 100000;
        
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
        box->AddValue (lat);
        box->AddValue (err_rate);
        box->AddValue (mincw);

        // reset every statistics
        m_agent_state[i]->Reset ();
        rman->GetAggInfo ().Reset ();
    }

    // for (NodeList::Iterator i = NodeList::Begin (); i != NodeList::End (); ++i) {
    //     Ptr<Node> node = *i;
    //     Ptr<WifiMacQueue> queue = GetQueue (node);
    //     uint32_t value = queue->GetNPackets();
    //     box->AddValue(value);
    // }

    NS_LOG_UNCOND ("MyGetObservation: " << box);
    return box;
}

float
MyGymEnv::GetReward()
{
    NS_LOG_FUNCTION (this);
    static float lastValue = 0.0;
    float reward = m_rxPktNum - lastValue;
    lastValue = m_rxPktNum;
    NS_LOG_UNCOND ("MyGetReward: " << reward);
    return reward;
}

std::string
MyGymEnv::GetExtraInfo()
{
    NS_LOG_FUNCTION (this);
    std::string myInfo = "currentNodeId";
    // myInfo += "=";
    // if (m_currentNode) {
    //     myInfo += std::to_string(m_currentNode->GetId());
    // }
    // NS_LOG_UNCOND("MyGetExtraInfo: " << myInfo);
    return myInfo;
}

bool
MyGymEnv::SetCw(Ptr<Node> node, uint32_t cwMinValue, uint32_t cwMaxValue)
{
    Ptr<NetDevice> dev = node->GetDevice (0);
    Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice> (dev);
    Ptr<WifiMac> wifi_mac = wifi_dev->GetMac ();
    Ptr<RegularWifiMac> rmac = DynamicCast<RegularWifiMac> (wifi_mac);
    PointerValue ptr;
    rmac->GetAttribute ("Txop", ptr);
    Ptr<Txop> txop = ptr.Get<Txop> ();

    // if both set to the same value then we have uniform backoff?
    if (cwMinValue != 0) {
        NS_LOG_DEBUG ("Set CW min: " << cwMinValue);
        txop->SetMinCw(cwMinValue);
    }

    if (cwMaxValue != 0) {
        NS_LOG_DEBUG ("Set CW max: " << cwMaxValue);
        txop->SetMaxCw(cwMaxValue);
    }

    return true;
}

bool
MyGymEnv::ExecuteActions(Ptr<OpenGymDataContainer> action)
{
    NS_LOG_FUNCTION (this);
    NS_LOG_UNCOND ("MyExecuteActions: " << action);
    Ptr<OpenGymBoxContainer<uint32_t>> box = DynamicCast<OpenGymBoxContainer<uint32_t>>(action);
    std::vector<uint32_t> actionVector = box->GetData();

    uint32_t agentNum = m_agents.GetN ();
    for (uint32_t i=0; i<agentNum; i++) {
        Ptr<Node> node = m_agents.Get (i);
        uint32_t cwSize = actionVector.at(i);
        SetCw(node, cwSize, cwSize);
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
MyGymEnv::CountRxPkts(Ptr<MyGymEnv> entity, Ptr<Node> node, Ptr<const Packet> packet)
{
    NS_LOG_DEBUG ("Client received a packet of " << packet->GetSize () << " bytes");
    // entity->m_currentNode = node;
    entity->m_rxPktNum++;

    // entity->m_delay_estimator->RecordRx (packet);
    
    // NS_LOG_DEBUG ("Node with ID " << node->GetId() 
    //             //    << " received " << entity->m_rxPktNum << " packets, "
    //                << " received packet " << packet->GetUid() << " with size " << packet->GetSize() << ", "
    //                << "estimated delay: " << entity->m_delay_estimator->GetLastDelay ());

}

// This method is called upon ACK receiving
void
MyGymEnv::SrcTxDone(Ptr<MyGymEnv> entity, Ptr<Node> node, uint32_t idx, const WifiMacHeader &hdr)
{
    Packet *packet = (Packet *) hdr.m_packet;



    // record TX throughput and latency
    Ptr<MyGymNodeState> state = entity->m_agent_state[idx];
    state->m_txPktCount++;
    state->m_txPktBytes += packet->GetSize ();
    state->m_delay_estimator->RecordRx (packet);
    state->m_delaySum += state->m_delay_estimator->GetLastDelay ();

    // entity->m_delay_estimator->RecordRx (packet);

    // NS_LOG_UNCOND ("Node with ID " << node->GetId()
    //                << " sent packet " << packet->GetUid() << " with size " << packet->GetSize() << ", "
    //                << "estimated delay: " << entity->m_delay_estimator->GetLastDelay () << ", "
    //                << "error rate: " << error_rate);
}

// This method is called upon Tx fail
void
MyGymEnv::SrcTxFail(Ptr<MyGymEnv> entity, Ptr<Node> node, uint32_t idx, const WifiMacHeader &hdr)
{
    Packet *packet = (Packet *) hdr.m_packet;

    NS_LOG_UNCOND ("Node with ID " << node->GetId() << " has failed to send a packet with size " << packet->GetSize());
}

} // ns3 namespace