#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/opengym-module.h"
#include "ns3/spectrum-module.h"
#include "ns3/wifi-module.h"
#include "utils.h"
#include "gym.h"

using namespace ns3;

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
  // initial position allocation
  Ptr<ListPositionAllocator> positionAllocator = CreateObject<ListPositionAllocator> ();
  for (auto it = graph->PosBegin (); it != graph->PosEnd (); it++)
    positionAllocator->Add (Vector (it->first, it->second, 0.0));
  mobility.SetPositionAllocator (positionAllocator);
  if (config.mobility == "fixed")
    mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  else if (config.mobility == "paired")
    {
      NS_ASSERT (config.loss == "geometric" && config.layout == "link");
      bool flag = false;
      for (auto i = nodes.Begin (); i != nodes.End (); i++)
        {
          if ((flag = !flag))
            {
              Ptr<RandomRectanglePositionAllocator> positionAllocator = CreateObject<RandomRectanglePositionAllocator> ();

              // randomly move according to graphSeed
              uint32_t idx = i - nodes.Begin ();
              positionAllocator->SetX (CreateObject<MyUniformRandomVariable> (config.graphSeed + idx * 3 + 1));
              positionAllocator->SetY (CreateObject<MyUniformRandomVariable> (config.graphSeed + idx * 3 + 2));
              Ptr<MyUniformRandomVariable> speed = CreateObject<MyUniformRandomVariable> (config.graphSeed + idx * 3 + 3, 0.3, 0.7);

              mobility.SetMobilityModel ("ns3::RandomWaypointMobilityModel", "PositionAllocator", PointerValue (positionAllocator),
                                                                             "Speed", PointerValue (speed));
              mobility.Install (*i);

              Ptr<MobilityModel> mobilityModel = (*i)->GetObject<MobilityModel> ();
              // set reference point for the next node
              mobility.PushReferenceMobilityModel (mobilityModel);
            }
          else
            {
              mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
              mobility.Install (*i);
              mobility.PopReferenceMobilityModel ();
            }
        }
    }
  else if (config.mobility == "random")
    {
      NS_ASSERT (config.loss == "geometric");
      for (auto i = nodes.Begin (); i != nodes.End (); i++)
        {
          Ptr<RandomRectanglePositionAllocator> positionAllocator = CreateObject<RandomRectanglePositionAllocator> ();
          // randomly move according to graphSeed
          uint32_t idx = i - nodes.Begin ();
          positionAllocator->SetX (CreateObject<MyUniformRandomVariable> (config.graphSeed + idx * 3 + 1));
          positionAllocator->SetY (CreateObject<MyUniformRandomVariable> (config.graphSeed + idx * 3 + 2));
          Ptr<MyUniformRandomVariable> speed = CreateObject<MyUniformRandomVariable> (config.graphSeed + idx * 3 + 3, 0.3, 0.7);

          mobility.SetMobilityModel ("ns3::RandomWaypointMobilityModel", "PositionAllocator", PointerValue (positionAllocator),
                                                                          "Speed", PointerValue (speed));
          mobility.Install (*i);
        }
    }
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
  if (config.loss == "graph")
    ConfigureMatrixTopology (spectrumChannel, graph, nodes);
  else if (config.loss == "geometric")
    {
      Ptr<RangePropagationLossModel> lossModel = CreateObject<RangePropagationLossModel> ();
      lossModel->SetAttribute ("MaxRange", DoubleValue (config.threshold));
      spectrumChannel->AddPropagationLossModel (lossModel);
    }

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

Ptr<GymEnv>
SetupOpenGym (NodeContainer &srcNodes, const MyConfig &config)
{
  Ptr<OpenGymInterface> openGymInterface = CreateObject<OpenGymInterface> (config.openGymPort);
  Ptr<GymEnv> gymEnv = CreateObject<GymEnv> (srcNodes, config);

  gymEnv->SetOpenGymInterface (openGymInterface);

  const uint32_t nFlows = srcNodes.GetN ();
  for (uint32_t i = 0; i < nFlows; i++)
    {
      Ptr<Node> node = srcNodes.Get (i);
      Ptr<NetDevice> dev = node->GetDevice (0);
      Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice> (dev);
      Ptr<WifiMac> wifi_mac = wifi_dev->GetMac ();
      Ptr<RegularWifiMac> rmac = DynamicCast<RegularWifiMac> (wifi_mac);

      rmac->TraceConnectWithoutContext (
        "TxOkHeader", MakeBoundCallback (&GymEnv::SrcTxDone, gymEnv, i));
      
      Ptr<WifiPhy> phy = rmac->GetWifiPhy ();
      PointerValue ptr;
      phy->GetAttribute ("State", ptr);
      Ptr<WifiPhyStateHelper> phy_state = ptr.Get<WifiPhyStateHelper> ();
      phy_state->TraceConnectWithoutContext (
        "State", MakeBoundCallback (&GymEnv::PhyStateChange, gymEnv, i));
    }

  return gymEnv;
}

void
CheckConfig (MyConfig &config)
{
  // argument check
  // MYNOTE: we can do this elsewhere
  if (config.layout != "node" && config.layout != "link")
    NS_FATAL_ERROR ("layout must be (node|link)");
  if (config.loss != "graph" && config.loss != "geometric")
    NS_FATAL_ERROR ("loss must be (graph|geometric)");
  if (config.mobility != "fixed" && config.mobility != "paired" && config.mobility != "random")
    NS_FATAL_ERROR ("mobility must be (fixed|paired|random)");

  // parse and check N,d
  // when topology==single is unchanged, just use default config
  if (config.loss == "geometric" && config.topology != "single")
    {
      std::stringstream ss (config.topology);
      std::string s;
      std::vector<float> params;
      while (getline (ss, s, ','))
        params.push_back (std::stof (s));

      // parameter check
      if (params.size () != 2)
        NS_FATAL_ERROR ("geometric configuration has two parameters");
      if (static_cast<uint32_t> (params[0]) != params[0])
        NS_FATAL_ERROR ("geometric configuration N has to be integer");
      config.nFlows = static_cast<uint32_t> (params[0]);
      if (params[1] < 0.0 || params[1] > 1.0)
        NS_FATAL_ERROR ("geometric configuration d has to be in [0, 1]");
      config.threshold = params[1];
    }
}


int
main (int argc, char *argv[])
{
  MyConfig config;

  CommandLine cmd;
  cmd.AddValue ("openGymPort", "Port number for OpenGym env.", config.openGymPort);
  cmd.AddValue ("simSeed", "Seed for random generator.", config.simSeed);
  cmd.AddValue ("simTime", "Simulation time in seconds,", config.simTime);
  cmd.AddValue ("stepTime", "Step time of the environment.", config.stepTime);

  cmd.AddValue ("graphSeed", "Seed for the graph generation.", config.graphSeed);
  cmd.AddValue ("layout", "Topology layout (link|node).", config.layout);
  cmd.AddValue ("loss", "Loss configuration (graph|geometric).", config.loss);
  cmd.AddValue ("topology", "Interference topology file (if loss==graph), geometric graph parameter N,d (if loss==geometric).",
                config.topology);

  cmd.AddValue ("mobility", "Mobility of the nodes (fixed|paired|random)", config.mobility);
  cmd.AddValue ("intensity", "Intensity of the traffic.", config.intensity);

  cmd.AddValue ("algorithm", "MAC algorithm to use (80211|odcf|rl).",
                config.algorithm);
  cmd.AddValue ("debug", "Print debug message.", config.debug);
  cmd.Parse (argc, argv);

  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (config.simSeed);

  Config::SetDefault ("ns3::WifiMacQueue::MaxDelay", TimeValue (Seconds (config.simTime)));

  CheckConfig (config);

  // graph read
  Ptr<Graph> graph = BuildGraph (config);

  // Create nodes
  NodeContainer nodes;
  nodes.Create (graph->GetNNodes ());

  NetDeviceContainer devices = SetupWifi (nodes, graph, config);
  SetupInternet (nodes, devices);
  NodeContainer srcNodes = SetupApplication (nodes, graph, config);

  Ptr<GymEnv> gymEnv = SetupOpenGym (srcNodes, config);

  Simulator::Stop (Seconds (config.simTime));
  Simulator::Run ();

  gymEnv->PrintResults ();
  gymEnv->NotifySimulationEnd ();

  Simulator::Destroy ();

  return EXIT_SUCCESS;
}