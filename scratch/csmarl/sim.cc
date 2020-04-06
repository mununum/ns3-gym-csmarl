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
#include "scenario.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("OpenGym");

int
main (int argc, char *argv[])
{

  // Parameters of the environment
  uint32_t simSeed = 1;
  double simulationTime = 20; // seconds
  double envStepTime = 0.02; // seconds, ns3gym env step time interval
  uint32_t openGymPort = 5555;
  bool debug = true;

  // OpenGym Env
  bool opengymEnabled = false;
  bool continuous = false;

  // Parameters of the scenario
  std::string topology = "fim";
  uint32_t nFlows = 3;
  double distance = 10.0;
  bool noErrors = false;
  std::string errorModelType = "ns3::NistErrorRateModel";
  // bool enableFading = true;
  uint32_t pktPerSec = 1000;
  uint32_t payloadSize = 1500; // 1500 B/pkt * 8 b/B * 1000 pkt/s = 12.0 Mbps
  bool enabledMinstrel = false;

  // define datarates
  std::vector<std::string> dataRates;
  dataRates.push_back ("OfdmRate1_5MbpsBW5MHz");
  dataRates.push_back ("OfdmRate2_25MbpsBW5MHz");
  dataRates.push_back ("OfdmRate3MbpsBW5MHz");
  dataRates.push_back ("OfdmRate4_5MbpsBW5MHz");
  dataRates.push_back ("OfdmRate6MbpsBW5MHz");
  dataRates.push_back ("OfdmRate9MbpsBW5MHz"); // <--
  dataRates.push_back ("OfdmRate12MbpsBW5MHz");
  dataRates.push_back ("OfdmRate13_5MbpsBW5MHz");
  uint32_t dataRateId = 5;

  CommandLine cmd;
  // required parameters for OpenGym interface
  cmd.AddValue ("openGymPort", "Port number for OpenGym env. Default: 5555", openGymPort);
  cmd.AddValue ("simSeed", "Seed for random generator. Default: 1", simSeed);
  // optional parameters
  cmd.AddValue ("simTime", "Simulation time in seconds, Default: 20s", simulationTime);
  cmd.AddValue ("stepTime", "Step time of the environment, Default: 0.02s", envStepTime);
  // MYTODO add topologies
  cmd.AddValue ("topology", "Interference topology. [fc, fim, ...], Default: fim", topology);
  cmd.AddValue ("nFlows", "Number of flows. Default: 3", nFlows);
  cmd.AddValue ("distance", "Inter node distance. Default: 10m", distance);
  cmd.AddValue ("opengymEnabled", "Using openAI gym or not. Default: false", opengymEnabled);
  cmd.AddValue ("continuous", "Use continuous action space. Default: false", continuous);
  cmd.AddValue ("debug", "Print debug message. Default: true", debug);
  cmd.Parse (argc, argv);

  NS_LOG_UNCOND ("Ns3Env parameters:");
  NS_LOG_UNCOND ("--simulationTime: " << simulationTime);
  NS_LOG_UNCOND ("--openGymPort: " << openGymPort);
  NS_LOG_UNCOND ("--envStepTime: " << envStepTime);
  NS_LOG_UNCOND ("--seed: " << simSeed);
  NS_LOG_UNCOND ("--distance: " << distance);

  if (debug)
    {
      LogComponentEnable ("OpenGym", LOG_LEVEL_DEBUG);
      LogComponentEnable ("MyGymEnv", LOG_LEVEL_DEBUG);
      LogComponentEnable ("OpenGymInterface", LOG_LEVEL_DEBUG);
    }

  if (noErrors)
    {
      errorModelType = "ns3::NoErrorRateModel";
    }

  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (simSeed);

  // Configuration of the scenario
  // Create Nodes
  uint32_t nodeNum = nFlows * 2;
  NodeContainer nodes;
  nodes.Create (nodeNum);

  // WiFi device
  WifiHelper wifi;
  wifi.SetStandard (WIFI_PHY_STANDARD_80211_5MHZ);

  // Mobility model
  MobilityHelper mobility;
  // mobility.SetPositionAllocator ("ns3::GridPositionAllocator", "MinX", DoubleValue (0.0), "MinY",
  //                                DoubleValue (0.0), "DeltaX", DoubleValue (distance), "DeltaY",
  //                                DoubleValue (distance), "GridWidth",
  //                                UintegerValue (2), // will create FIM topology
  //                                "LayoutType", StringValue ("RowFirst"));
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  for (uint32_t i = 0; i < nFlows; i++)
    {
      // The nodes will be overlapped
      positionAlloc->Add (Vector (0.0, 0.0, 0.0));
      positionAlloc->Add (Vector (5.0, 0.0, 0.0));
    }
  mobility.SetPositionAllocator (positionAlloc);

  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (nodes);

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
  // Ptr<FriisPropagationLossModel> lossModel = CreateObject<FriisPropagationLossModel> ();
  // Ptr<NakagamiPropagationLossModel> fadingModel = CreateObject<NakagamiPropagationLossModel> ();
  // if (enableFading)
  //   {
  //     lossModel->SetNext (fadingModel);
  //   }
  // spectrumChannel->AddPropagationLossModel (lossModel);

  if (topology == "fc")
    {
      ConfigureFCTopology (spectrumChannel, nodes);
    }
  else if (topology == "fim")
    {
      ConfigureFIMTopology (spectrumChannel, nodes);
    }
  else
    {
      NS_LOG_ERROR ("invalid topology configuration");
      exit (1);
    }

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

  // Send a UDP trafic from even --> odd

  uint16_t port = 1000;
  // uint32_t srcNodeId = 0;
  // uint32_t destNodeId = 1;
  NodeContainer agents;
  ApplicationContainer sourceApps;
  ApplicationContainer sinkApps;
  for (uint32_t srcNodeId = 0; srcNodeId < nodeNum; srcNodeId += 2)
    {
      uint32_t destNodeId = srcNodeId + 1;
      Ptr<Node> srcNode = nodes.Get (srcNodeId);
      Ptr<Node> dstNode = nodes.Get (destNodeId);

      agents.Add (srcNode);

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

      // RandomAppHelper source ("ns3::UdpSocketFactory", InetSocketAddress (destAddress));
      // source.SetAttribute ("Delay", StringValue ("ns3::ExponentialRandomVariable[Mean=" + std::to_string(1.0 / pktPerSec) + "]"));
      // source.SetAttribute ("Size", StringValue ("ns3::ExponentialRandomVariable[Mean=" + std::to_string(payloadSize) + "][Bound=2000]"));

      ApplicationContainer newSourceApps = source.Install (srcNode);
      sourceApps.Add (newSourceApps);

      // Create a packet sink to receive these packets
      UdpServerHelper sink (port);
      ApplicationContainer newSinkApps = sink.Install (dstNode);
      sinkApps.Add (newSinkApps);
    }

  sourceApps.Start (Seconds (0.0));
  sourceApps.Stop (Seconds (simulationTime));
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

  // Configure OpenGym environment
  Ptr<OpenGymInterface> openGymInterface = CreateObject<OpenGymInterface> (openGymPort);
  Ptr<MyGymEnv> myGymEnv =
      CreateObject<MyGymEnv> (agents, Seconds (simulationTime), Seconds (envStepTime),
                              opengymEnabled, continuous);

  myGymEnv->SetOpenGymInterface (openGymInterface);

  // connect OpenGym entity to RX event source

  for (uint32_t i = 0; i < nFlows; i++)
    {
      Ptr<UdpServer> udpServer = DynamicCast<UdpServer> (sinkApps.Get (i));
      Ptr<Node> dstNode = nodes.Get (i * 2 + 1);
      udpServer->TraceConnectWithoutContext (
          "Rx", MakeBoundCallback (&MyGymEnv::CountRxPkts, myGymEnv, dstNode, i));
    }
  // udpServer->TraceConnectWithoutContext ("Rx", MakeCallback (&DestRxPkt));

  // connect TxOkHeader trace source
  for (uint32_t i = 0; i < nFlows; i++)
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

  // NS_LOG_UNCOND (myGymEnv->GetTotalPkt ());
  std::cout << myGymEnv->GetTotalPkt () << std::endl;

  openGymInterface->NotifySimulationEnd ();

  Simulator::Destroy ();
}