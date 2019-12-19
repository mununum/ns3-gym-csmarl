#ifndef MY_GYM_ENTITY_H
#define MY_GYM_ENTITY_H

#include "ns3/node-container.h"
#include "ns3/wifi-mac-header.h"
#include "ns3/stats-module.h"
#include "ns3/opengym-module.h"
#include "ns3/delay-jitter-estimation.h"

namespace ns3 {

class Node;
class WifiMacQueue;
class Packet;
class MyGymNodeState;

class MyGymEnv : public OpenGymEnv
{
public:
    MyGymEnv (); // for typeid registration
    MyGymEnv (NodeContainer agents, Time stepTime, bool enabled);

    virtual ~MyGymEnv ();
    static TypeId GetTypeId (void);
    virtual void DoDispose ();

    Ptr<OpenGymSpace> GetActionSpace();
    Ptr<OpenGymSpace> GetObservationSpace();
    bool GetGameOver();
    Ptr<OpenGymDataContainer> GetObservation();
    float GetReward();
    std::string GetExtraInfo();
    bool ExecuteActions(Ptr<OpenGymDataContainer> action);

    // the function has to be static to work with MakeBoundCallback
    // that is why we pass pointer to MyGymEnv instance to be able to store the context (node, etc)
    // static void NotifyPktRxEvent(Ptr<MyGymEnv> entity, Ptr<Node> node, Ptr<const Packet> packet); // for event-based env
    static void CountRxPkts(Ptr<MyGymEnv> entity, Ptr<Node> node, Ptr<const Packet> packet);
    static void SrcTxDone(Ptr<MyGymEnv> entity, Ptr<Node> node, uint32_t idx, const WifiMacHeader &hdr);
    static void SrcTxFail(Ptr<MyGymEnv> entity, Ptr<Node> node, uint32_t idx, const WifiMacHeader &hdr);

private:
    void ScheduleNextStateRead();
    Ptr<WifiMacQueue> GetQueue(Ptr<Node> node);
    bool SetCw(Ptr<Node> node, uint32_t cwMinValue=0, uint32_t cwMaxValue=0);

    NodeContainer m_agents;
    std::vector<Ptr<MyGymNodeState> > m_agent_state;

    std::vector<uint32_t> m_obs_shape;
    Time m_interval = Seconds(0.1);
    // Ptr<Node> m_currentNode;
    uint64_t m_rxPktNum;

    bool m_enabled;
    bool m_continuous;
    // Ptr<DelayJitterEstimation> m_delay_estimator;

};

}

#endif // MY_GYM_ENTITY_H