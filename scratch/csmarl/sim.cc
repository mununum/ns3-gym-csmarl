#include "ns3/core-module.h"
#include "ns3/applications-module.h"
#include "ns3/opengym-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/spectrum-module.h"
#include "ns3/stats-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/traffic-control-module.h"
#include "ns3/node-list.h"

#include "mygym.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("OpenGym");

int
main (int argc, char *argv[])
{

  // LogComponentEnable("Txop", LOG_LEVEL_DEBUG);
  // LogComponentEnable("ArpL3Protocol", LOG_LEVEL_LOGIC);
  // LogComponentEnable("OpenGymInterface", LOG_LEVEL_DEBUG);
  // LogComponentEnable("ChannelAccessManager", LOG_LEVEL_DEBUG);

  // Parameters of the environment
  uint32_t simSeed = 1;
  double simulationTime = 10; // seconds
  double envStepTime = 0.1; // seconds, ns3gym env step time interval
  uint32_t openGymPort = 5555;
  // uint32_t testArg = 0;

  // OpenGym Env
  bool opengymEnabled = true;
  bool continuous = false;
  bool dynamicInterval = true;

  // Parameters of the scenario
  uint32_t nodeNum = 2;
  double distance = 10.0;
  bool noErrors = false;
  std::string errorModelType = "ns3::NistErrorRateModel";
  bool enableFading = true;
  uint32_t pktPerSec = 1000;
  uint32_t payloadSize = 1500;
  bool enabledMinstrel = false;

  // define datarates
  std::vector<std::string> dataRates;
  dataRates.push_back ("OfdmRate1_5MbpsBW5MHz");
  dataRates.push_back ("OfdmRate2_25MbpsBW5MHz");
  dataRates.push_back ("OfdmRate3MbpsBW5MHz");
  dataRates.push_back ("OfdmRate4_5MbpsBW5MHz");
  dataRates.push_back ("OfdmRate6MbpsBW5MHz");
  dataRates.push_back ("OfdmRate9MbpsBW5MHz");
  dataRates.push_back ("OfdmRate12MbpsBW5MHz");
  dataRates.push_back ("OfdmRate13_5MbpsBW5MHz");
  uint32_t dataRateId = 5;

  CommandLine cmd;
  // required parameters for OpenGym interface
  cmd.AddValue ("openGymPort", "Port number for OpenGym env. Default: 5555", openGymPort);
  cmd.AddValue ("simSeed", "Seed for random generator. Default: 1", simSeed);
  // optional parameters
  cmd.AddValue ("simTime", "Simulation time in seconds, Default: 10s", simulationTime);
  cmd.AddValue ("stepTime", "Step time of the environment, Default: 0.1s", envStepTime);
  cmd.AddValue ("nodeNum", "Number of nodes. Default: 2", nodeNum);
  cmd.AddValue ("distance", "Inter node distance. Default: 10m", distance);
  cmd.AddValue ("opengymEnabled", "Using openAI gym or not. Default: true", opengymEnabled);
  cmd.AddValue ("continuous", "Use continuous action space. Default: false", continuous);
  cmd.AddValue ("dynamicInterval", "Dynamically changing step interval. Default: true",
                dynamicInterval);
  // cmd.AddValue ("testArg", "Extra simulation argument. Default: 0", testArg);
  cmd.Parse (argc, argv);

  NS_LOG_UNCOND ("Ns3Env parameters:");
  NS_LOG_UNCOND ("--simulationTime: " << simulationTime);
  NS_LOG_UNCOND ("--openGymPort: " << openGymPort);
  NS_LOG_UNCOND ("--envStepTime: " << envStepTime);
  NS_LOG_UNCOND ("--seed: " << simSeed);
  NS_LOG_UNCOND ("--distance: " << distance);
  // NS_LOG_UNCOND ("--testArg: " << testArg);

  if (noErrors)
    {
      errorModelType = "ns3::NoErrorRateModel";
    }

  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (simSeed);

  // Configuration of the scenario
  // Create Nodes
  NodeContainer nodes;
  nodes.Create (nodeNum);

  // WiFi device
  WifiHelper wifi;
  wifi.SetStandard (WIFI_PHY_STANDARD_80211_5MHZ);

  // Channel
  SpectrumWifiPhyHelper spectrumPhy = SpectrumWifiPhyHelper::Default ();
  Ptr<MultiModelSpectrumChannel> spectrumChannel = CreateObject<MultiModelSpectrumChannel> ();

  spectrumPhy.SetChannel (spectrumChannel);
  spectrumPhy.SetErrorRateModel (errorModelType);
  spectrumPhy.Set ("Frequency", UintegerValue (5200));
  spectrumPhy.Set ("ChannelWidth", UintegerValue (5));
  spectrumPhy.Set ("ShortGuardEnabled", BooleanValue (false));

  Config::SetDefault ("ns3::WifiPhy::CcaMode1Threshold", DoubleValue (-82.0));
  Config::SetDefault ("ns3::WifiPhy::Frequency", UintegerValue (5200));
  Config::SetDefault ("ns3::WifiPhy::ChannelWidth", UintegerValue (5));

  // Channel
  Ptr<FriisPropagationLossModel> lossModel = CreateObject<FriisPropagationLossModel> ();
  Ptr<NakagamiPropagationLossModel> fadingModel = CreateObject<NakagamiPropagationLossModel> ();
  if (enableFading)
    {
      lossModel->SetNext (fadingModel);
    }
  spectrumChannel->AddPropagationLossModel (lossModel);
  Ptr<ConstantSpeedPropagationDelayModel> delayModel =
      CreateObject<ConstantSpeedPropagationDelayModel> ();
  spectrumChannel->SetPropagationDelayModel (delayModel);

  // Add MAC and set DataRate
  WifiMacHelper wifiMac;

  if (enabledMinstrel)
    {
      wifi.SetRemoteStationManager ("ns3::MinstrelWifiManager");
    }
  else
    {
      std::string dataRateStr = dataRates.at (dataRateId);
      NS_LOG_UNCOND ("dataRateStr:" << dataRateStr);
      wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager", "DataMode",
                                    StringValue (dataRateStr), "ControlMode",
                                    StringValue (dataRateStr));
    }

  // Set it to adhoc mode
  wifiMac.SetType ("ns3::AdhocWifiMac", "QosSupported", BooleanValue (false));

  // Install wifi device
  NetDeviceContainer devices = wifi.Install (spectrumPhy, wifiMac, nodes);

  // Mobility model
  MobilityHelper mobility;
  mobility.SetPositionAllocator ("ns3::GridPositionAllocator", "MinX", DoubleValue (0.0), "MinY",
                                 DoubleValue (0.0), "DeltaX", DoubleValue (distance), "DeltaY",
                                 DoubleValue (distance), "GridWidth",
                                 UintegerValue (nodeNum), // will create linear topology
                                 "LayoutType", StringValue ("RowFirst"));
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (nodes);

  // IP stack and routing
  InternetStackHelper internet;
  internet.Install (nodes);

  // Assign IP addresses to devices
  Ipv4AddressHelper ipv4;
  NS_LOG_INFO ("Assign IP Addresses");
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces = ipv4.Assign (devices);

  // Configure routing (no need right now)

  // Traffic
  // Create a BulkSendApplication and install it on node 0
  // Ptr<UniformRandomVariable> startTimeRng = CreateObject<UniformRandomVariable> ();
  // startTimeRng->SetAttribute ("Min", DoubleValue (0.0));
  // startTimeRng->SetAttribute ("Max", DoubleValue (1.0));

  // Send a UDP trafic from 0 --> 1

  uint16_t port = 1000;
  uint32_t srcNodeId = 0;
  uint32_t destNodeId = 1;
  Ptr<Node> srcNode = nodes.Get (srcNodeId);
  Ptr<Node> dstNode = nodes.Get (destNodeId);

  Ptr<Ipv4> destIpv4 = dstNode->GetObject<Ipv4> ();
  Ipv4InterfaceAddress dest_ipv4_int_addr = destIpv4->GetAddress (1, 0);
  Ipv4Address dest_ip_addr = dest_ipv4_int_addr.GetLocal ();

  InetSocketAddress destAddress (dest_ip_addr, port);
  destAddress.SetTos (0x70); // AC_BE
  UdpClientHelper source (destAddress);
  source.SetAttribute ("MaxPackets", UintegerValue (pktPerSec * simulationTime));
  source.SetAttribute ("PacketSize", UintegerValue (payloadSize));
  Time interPacketInterval = Seconds (1.0 / pktPerSec);
  source.SetAttribute ("Interval", TimeValue (interPacketInterval)); // packets/s

  ApplicationContainer sourceApps = source.Install (srcNode);
  sourceApps.Start (Seconds (0.0));
  sourceApps.Stop (Seconds (simulationTime));

  // Create a packet sink to receive these packets
  UdpServerHelper sink (port);
  ApplicationContainer sinkApps = sink.Install (dstNode);
  sinkApps.Start (Seconds (0.0));
  sinkApps.Stop (Seconds (simulationTime));

  // Print node positions
  NS_LOG_UNCOND ("Node Positions:");
  for (uint32_t i = 0; i < nodes.GetN (); i++)
    {
      Ptr<Node> node = nodes.Get (i);
      Ptr<MobilityModel> mobility = node->GetObject<MobilityModel> ();
      NS_LOG_UNCOND ("---Node ID: " << node->GetId ()
                                    << " Positions: " << mobility->GetPosition ());
    }

  // Ptr<NodeContainer> agents = CreateObject<NodeContainer> ();
  NodeContainer agents;
  agents.Add (srcNode);

  Ptr<OpenGymInterface> openGymInterface = CreateObject<OpenGymInterface> (openGymPort);
  Ptr<MyGymEnv> myGymEnv =
      CreateObject<MyGymEnv> (agents, Seconds (envStepTime), opengymEnabled, continuous, dynamicInterval);

  myGymEnv->SetOpenGymInterface (openGymInterface);

  // connect OpenGym entity to RX event source
  Ptr<UdpServer> udpServer = DynamicCast<UdpServer> (sinkApps.Get (0));
  // udpServer->TraceConnectWithoutContext ("Rx", MakeCallback (&DestRxPkt));
  udpServer->TraceConnectWithoutContext (
      "Rx", MakeBoundCallback (&MyGymEnv::CountRxPkts, myGymEnv, dstNode));

  // connect TxOkHeader trace source
  uint32_t numAgents = agents.GetN ();
  for (uint32_t i = 0; i < numAgents; i++)
    {
      Ptr<Node> node = agents.Get (i);
      Ptr<NetDevice> dev = node->GetDevice (0);
      Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice> (dev);
      Ptr<WifiMac> wifi_mac = wifi_dev->GetMac ();
      Ptr<RegularWifiMac> rmac = DynamicCast<RegularWifiMac> (wifi_mac);
      rmac->TraceConnectWithoutContext (
          "TxOkHeader", MakeBoundCallback (&MyGymEnv::SrcTxDone, myGymEnv, node, i));
      rmac->TraceConnectWithoutContext (
          "TxErrHeader", MakeBoundCallback (&MyGymEnv::SrcTxFail, myGymEnv, node, i));
    }

  NS_LOG_UNCOND ("Simulation start");
  Simulator::Stop (Seconds (simulationTime));
  Simulator::Run ();
  NS_LOG_UNCOND ("Simulation stop");

  openGymInterface->NotifySimulationEnd ();

  Simulator::Destroy ();
}