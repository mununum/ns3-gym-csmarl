#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/opengym-module.h"
#include "ns3/spectrum-module.h"
#include "ns3/wifi-module.h"
#include "utils.h"
#include "mygym.h"

using namespace ns3;

namespace ns3 {
class MyConfig
{
public:
  uint32_t openGymPort = 5555;
  uint32_t simSeed = 0;
  double simTime = 20;
  double stepTime = 0.005;
  std::string topology = "single";
  std::string algorithm = "80211";
  bool debug = false;
  double intensity = 1.0;
};
} // namespace ns3

NetDeviceContainer
SetupWifi (NodeContainer &nodes, const Ptr<const Graph> graph, const MyConfig &config)
{
  WifiHelper wifi;
  wifi.SetStandard (WIFI_PHY_STANDARD_80211_5MHZ);

  // Datarate
  std::string dataRateStr = "OfdmRate12MbpsBW5MHz";
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager", "DataMode",
                                StringValue (dataRateStr), "ControlMode",
                                StringValue (dataRateStr));

  // Mobility
  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAllocator = CreateObject<ListPositionAllocator> ();
  for (auto it = graph->PosBegin (); it != graph->PosEnd (); it++)
    positionAllocator->Add (Vector (it->first, it->second, 0.0));
  mobility.SetPositionAllocator (positionAllocator);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (nodes);

  // Channel
  SpectrumWifiPhyHelper spectrumPhy = SpectrumWifiPhyHelper::Default ();
  Ptr<MultiModelSpectrumChannel> spectrumChannel = CreateObject<MultiModelSpectrumChannel> ();
  spectrumPhy.SetChannel (spectrumChannel);
  spectrumPhy.SetErrorRateModel ("ns3::NistErrorRateModel");
  spectrumPhy.Set ("Frequency", UintegerValue (5200));
  spectrumPhy.Set ("ChannelWidth", UintegerValue (5));
  spectrumPhy.Set ("ShortGuardEnabled", BooleanValue (false));
  Config::SetDefault ("ns3::WifiPhy::CcaMode1Threshold", DoubleValue (-82.0));
  Config::SetDefault ("ns3::WifiPhy::Frequency", UintegerValue (5200));
  Config::SetDefault ("ns3::WifiPhy::ChannelWidth", UintegerValue (5));
  // Channel delay
  Ptr<ConstantSpeedPropagationDelayModel> delayModel =
      CreateObject<ConstantSpeedPropagationDelayModel> ();
  spectrumChannel->SetPropagationDelayModel (delayModel);
  // Channel loss
  ConfigureMatrixTopology (spectrumChannel, graph, nodes);

  // MAC algorithm
  WifiMacHelper wifiMac;
  if (config.algorithm == "80211")
    wifiMac.SetType ("ns3::AdhocWifiMac", "QosSupported", BooleanValue (false));
  else if (config.algorithm == "odcf" || config.algorithm == "rl")
    {
      wifiMac.SetType ("ns3::ODcfAdhocWifiMac", "QosSupported", BooleanValue (false));
      Config::SetDefault ("ns3::QueueBase::MaxSize",
                          QueueSizeValue (QueueSize ("1p"))); // IFQ length = 1
      if (config.algorithm == "rl")
        Config::SetDefault ("ns3::ODcf::RL_mode", BooleanValue (true));
    }
  else
    NS_FATAL_ERROR ("algorithm must be (80211|odcf|rl)");

  return wifi.Install (spectrumPhy, wifiMac, nodes);
}

void
SetupInternet (NodeContainer &nodes, NetDeviceContainer &devices)
{
  // IP stack and routing
  InternetStackHelper internet;
  internet.Install (nodes);

  // Assign IP addresses to devices
  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces = ipv4.Assign (devices);

  // Configure ARP cache
  PopulateArpCache ();
}

NodeContainer
SetupApplication (NodeContainer &nodes, const Ptr<const Graph> graph, const MyConfig &config)
{
  const uint16_t port = 1000;
  const uint32_t pktPerSecBase = 1000;
  const uint32_t pktPerSec = pktPerSecBase * config.intensity;
  const uint32_t payloadSize = 1500;

  NodeContainer srcNodes;
  NodeContainer sinkNodes;
  ApplicationContainer srcApps;
  ApplicationContainer sinkApps;
  for (auto it = graph->FlowBegin (); it != graph->FlowEnd (); it++)
    {
      Ptr<Node> srcNode = nodes.Get (it->first);
      Ptr<Node> sinkNode = nodes.Get (it->second);

      srcNodes.Add (srcNode);
      sinkNodes.Add (sinkNode);

      Ptr<Ipv4> destIpv4 = sinkNode->GetObject<Ipv4> ();
      Ipv4InterfaceAddress destIpv4IntAddr = destIpv4->GetAddress (1, 0);
      Ipv4Address destIpAddr = destIpv4IntAddr.GetLocal ();

      InetSocketAddress destAddress (destIpAddr, port);
      destAddress.SetTos (0x70); // AC_BE

      UdpClientHelper source (destAddress);
      source.SetAttribute ("MaxPackets", UintegerValue (pktPerSec * config.simTime));
      source.SetAttribute ("PacketSize", UintegerValue (payloadSize));
      Time interPacketInterval = Seconds (1.0 / pktPerSec);
      source.SetAttribute ("Interval", TimeValue (interPacketInterval)); // packets/s
      ApplicationContainer newSrcApps = source.Install (srcNode);
      srcApps.Add (newSrcApps);

      UdpServerHelper sink (port);
      ApplicationContainer newSinkApps = sink.Install (sinkNode);
    }

  srcApps.Start (Seconds (0.0));
  srcApps.Stop (Seconds (config.simTime));
  sinkApps.Start (Seconds (0.0));
  sinkApps.Stop (Seconds (config.simTime));

  return srcNodes;
}

Ptr<MyGymEnv>
SetupOpenGym (NodeContainer &srcNodes, const MyConfig &config)
{
  Ptr<OpenGymInterface> openGymInterface = CreateObject<OpenGymInterface> (config.openGymPort);
  Ptr<MyGymEnv> myGymEnv = CreateObject<MyGymEnv> (srcNodes, Seconds (config.stepTime), config.algorithm, config.debug);

  myGymEnv->SetOpenGymInterface (openGymInterface);

  const uint32_t nFlows = srcNodes.GetN ();
  for (uint32_t i = 0; i < nFlows; i++)
    {
      Ptr<Node> node = srcNodes.Get (i);
      Ptr<NetDevice> dev = node->GetDevice (0);
      Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice> (dev);
      Ptr<WifiMac> wifi_mac = wifi_dev->GetMac ();
      Ptr<RegularWifiMac> rmac = DynamicCast<RegularWifiMac> (wifi_mac);

      rmac->TraceConnectWithoutContext (
        "TxOkHeader", MakeBoundCallback (&MyGymEnv::SrcTxDone, myGymEnv, i));
      
      Ptr<WifiPhy> phy = rmac->GetWifiPhy ();
      PointerValue ptr;
      phy->GetAttribute ("State", ptr);
      Ptr<WifiPhyStateHelper> phy_state = ptr.Get<WifiPhyStateHelper> ();
      phy_state->TraceConnectWithoutContext (
        "State", MakeBoundCallback (&MyGymEnv::PhyStateChange, myGymEnv, i));
    }

  return myGymEnv;
}


int
main (int argc, char *argv[])
{
  MyConfig config;

  CommandLine cmd;
  cmd.AddValue ("openGymPort", "Port number for OpenGym env. Default: 5555", config.openGymPort);
  cmd.AddValue ("simSeed", "Seed for random generator. Default: 0", config.simSeed);
  cmd.AddValue ("simTime", "Simulation time in seconds, Default: 20s", config.simTime);
  cmd.AddValue ("stepTime", "Step time of the environment, Default: 0.005s", config.stepTime);
  cmd.AddValue ("topology", "Interference topology. (on graph file), Default: fim",
                config.topology);
  cmd.AddValue ("algorithm", "MAC algorithm to use (80211|odcf|rl). Default: 80211",
                config.algorithm);
  cmd.AddValue ("debug", "Print debug message. Default: false", config.debug);
  cmd.AddValue ("intensity", "Intensity of the traffic. Default: 1.0", config.intensity);
  cmd.Parse (argc, argv);

  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (config.simSeed);

  Config::SetDefault ("ns3::WifiMacQueue::MaxDelay", TimeValue (Seconds (config.simTime)));

  Ptr<Graph> graph = ReadLinkGraph (config.topology);

  // Create nodes
  NodeContainer nodes;
  nodes.Create (graph->GetNNodes ());

  NetDeviceContainer devices = SetupWifi (nodes, graph, config);
  SetupInternet (nodes, devices);
  NodeContainer srcNodes = SetupApplication (nodes, graph, config);

  Ptr<MyGymEnv> myGymEnv = SetupOpenGym (srcNodes, config);

  Simulator::Stop (Seconds (config.simTime));
  Simulator::Run ();

  myGymEnv->PrintResults ();
  myGymEnv->NotifySimulationEnd ();

  Simulator::Destroy ();

  return EXIT_SUCCESS;
}