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

void
PopulateArpCache ()
{
  // return;
  Ptr<ArpCache> arp = CreateObject<ArpCache> ();
  for (NodeList::Iterator i = NodeList::Begin (); i != NodeList::End (); i++)
    {
      Ptr<Ipv4L3Protocol> ip = (*i)->GetObject<Ipv4L3Protocol> ();
      NS_ASSERT (ip != 0);
      ObjectVectorValue interfaces;
      ip->GetAttribute ("InterfaceList", interfaces);
      for (ObjectVectorValue::Iterator j = interfaces.Begin (); j != interfaces.End (); j++)
        {
          Ptr<Ipv4Interface> ipIface = (j->second)->GetObject<Ipv4Interface> ();
          NS_ASSERT (ipIface != 0);
          Ptr<NetDevice> device = ipIface->GetDevice ();
          NS_ASSERT (device != 0);
          Mac48Address addr = Mac48Address::ConvertFrom (device->GetAddress ());
          for (uint32_t k = 0; k < ipIface->GetNAddresses (); k++)
            {
              Ipv4Address ipAddr = ipIface->GetAddress (k).GetLocal ();
              if (ipAddr == Ipv4Address::GetLoopback ())
                continue;
              ArpCache::Entry *entry = arp->Add (ipAddr);
              entry->SetMacAddress (addr);
              entry->MarkPermanent ();
            }
        }
    }
  for (NodeList::Iterator i = NodeList::Begin (); i != NodeList::End (); i++)
    {
      Ptr<Ipv4L3Protocol> ip = (*i)->GetObject<Ipv4L3Protocol> ();
      NS_ASSERT (ip != 0);
      ObjectVectorValue interfaces;
      ip->GetAttribute ("InterfaceList", interfaces);
      for (ObjectVectorValue::Iterator j = interfaces.Begin (); j != interfaces.End (); j++)
        {
          Ptr<Ipv4Interface> ipIface = (j->second)->GetObject<Ipv4Interface> ();
          ipIface->SetAttribute ("ArpCache", PointerValue (arp));
        }
    }
}

int
main (int argc, char *argv[])
{

  // Parameters of the environment
  uint32_t simSeed = 0;
  double simulationTime = 20; // seconds
  double envStepTime = 0.02; // seconds, ns3gym env step time interval
  uint32_t openGymPort = 5555;
  bool debug = true;

  // OpenGym Env
  std::string algorithm = "80211";
  bool continuous = false;

  // Parameters of the scenario
  std::string topology = "fim";
  std::string traffic = "cbr";
  bool noErrors = false;
  std::string errorModelType = "ns3::NistErrorRateModel";
  // bool enableFading = true;
  double intensity = 1.0;  // traffic intensity
  uint32_t pktPerSecBase = 1000;  // when intensity = 1: 1500 B/pkt * 8 b/B * 1000 pkt/s = 12.0 Mbps -- saturated traffic
  uint32_t payloadSize = 1500;
  bool enabledMinstrel = false;

  bool randomFlow = false;

  std::string queueSize = "100p";
  float delayRewardWeight = 0.0;

  // define datarates
  std::vector<std::string> dataRates;
  dataRates.push_back ("OfdmRate1_5MbpsBW5MHz");
  dataRates.push_back ("OfdmRate2_25MbpsBW5MHz");
  dataRates.push_back ("OfdmRate3MbpsBW5MHz");
  dataRates.push_back ("OfdmRate4_5MbpsBW5MHz");
  dataRates.push_back ("OfdmRate6MbpsBW5MHz");
  dataRates.push_back ("OfdmRate9MbpsBW5MHz");
  dataRates.push_back ("OfdmRate12MbpsBW5MHz"); // <--
  dataRates.push_back ("OfdmRate13_5MbpsBW5MHz");
  uint32_t dataRateId = 6;

  CommandLine cmd;
  // required parameters for OpenGym interface
  cmd.AddValue ("openGymPort", "Port number for OpenGym env. Default: 5555", openGymPort);
  cmd.AddValue ("simSeed", "Seed for random generator. Default: 0", simSeed);
  // optional parameters
  cmd.AddValue ("simTime", "Simulation time in seconds, Default: 20s", simulationTime);
  cmd.AddValue ("stepTime", "Step time of the environment, Default: 0.02s", envStepTime);
  cmd.AddValue ("topology", "Interference topology. (on graph file), Default: fim", topology);
  cmd.AddValue ("algorithm", "MAC algorithm to use (80211|odcf|rl). Defaule: 80211", algorithm);
  cmd.AddValue ("continuous", "Use continuous action space. Default: false", continuous);
  cmd.AddValue ("debug", "Print debug message. Default: true", debug);
  cmd.AddValue ("traffic", "Traffic type (cbr|mmpp). Default: cbr", traffic);
  cmd.AddValue ("intensity", "Intensity of the traffic. Default: 1.0", intensity);
  cmd.AddValue ("randomFlow", "Randomize flows. Default: false", randomFlow);
  cmd.AddValue ("queueSize", "Size of MAC layer buffer. Default: 100p", queueSize);
  cmd.AddValue ("delayRewardWeight", "Weight of delay reward. Default: 0.0", delayRewardWeight);
  cmd.Parse (argc, argv);

  NS_LOG_UNCOND ("Ns3Env parameters:");
  NS_LOG_UNCOND ("--simulationTime: " << simulationTime);
  NS_LOG_UNCOND ("--openGymPort: " << openGymPort);
  NS_LOG_UNCOND ("--envStepTime: " << envStepTime);
  NS_LOG_UNCOND ("--seed: " << simSeed);

  if (debug)
    {
      // LogComponentEnable ("OpenGym", LOG_LEVEL_DEBUG);
      // LogComponentEnable ("MyGymEnv", LOG_LEVEL_DEBUG);
      // LogComponentEnable ("OpenGymInterface", LOG_LEVEL_DEBUG);
      // LogComponentEnable ("ODcfQueue", LOG_LOGIC);
      // LogComponentEnable ("Queue", LOG_LEVEL_INFO);
    }

  if (noErrors)
    {
      errorModelType = "ns3::NoErrorRateModel";
    }

  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (simSeed);
  std::srand (simSeed);

  // open graph file
  uint32_t nNodes, nEdges, nFlows;
  std::vector<std::tuple<float, float>> pos;
  std::vector<std::tuple<uint32_t, uint32_t>> edges;
  std::vector<std::tuple<uint32_t, uint32_t>> flows;

  // read a graph file and assign values to nNodes, nEdges, nFlows
  // pos, edges, flows
  ReadGraph (topology, nNodes, nEdges, nFlows, pos, edges, flows);

  // When enabled, generate randomized flow for this example.
  if (randomFlow)
    MakeFlows (nNodes, nEdges, nFlows, edges, flows, simSeed);

  // Configuration of the scenario
  // Create Nodes
  NodeContainer nodes;
  nodes.Create (nNodes);

  // WiFi device
  WifiHelper wifi;
  wifi.SetStandard (WIFI_PHY_STANDARD_80211_5MHZ);

  // Mobility model
  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  for (uint32_t i = 0; i < nNodes; i++)
    {
      positionAlloc->Add (Vector (std::get<0> (pos[i]), std::get<1> (pos[i]), 0.0));
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

  // if (topology == "fc")
  //   {
  //     ConfigureFCTopology (spectrumChannel, nodes);
  //   }
  // else if (topology == "fim")
  //   {
  //     ConfigureFIMTopology (spectrumChannel, nodes);
  //   }
  // else
  //   {
  //     NS_FATAL_ERROR ("invalid topology configuration");
  //   }

  ConfigureMatrixTopology (spectrumChannel, nodes, nEdges, edges);

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
  Config::SetDefault ("ns3::WifiMacQueue::MaxDelay", TimeValue (Seconds (simulationTime)));
  Config::SetDefault ("ns3::QueueBase::MaxSize", QueueSizeValue (QueueSize (queueSize)));
  if (algorithm == "odcf")
    wifiMac.SetType ("ns3::ODcfAdhocWifiMac", "QosSupported", BooleanValue (false));
  else if (algorithm == "80211" || algorithm == "rl")
    wifiMac.SetType ("ns3::AdhocWifiMac", "QosSupported", BooleanValue (false));
  else
    NS_FATAL_ERROR ("algorithm must be (80211|odcf|rl)");
  

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

  // Configure ARP cache
  PopulateArpCache ();

  uint16_t port = 1000;
  NodeContainer srcNodes;
  NodeContainer sinkNodes;
  ApplicationContainer sourceApps;
  ApplicationContainer sinkApps;
  for (uint32_t i = 0; i < nFlows; i++)
    {
      uint32_t srcNodeId = std::get<0> (flows[i]);
      uint32_t sinkNodeId = std::get<1> (flows[i]);

      Ptr<Node> srcNode = nodes.Get (srcNodeId);
      Ptr<Node> sinkNode = nodes.Get (sinkNodeId);

      srcNodes.Add (srcNode);

      Ptr<Ipv4> destIpv4 = sinkNode->GetObject<Ipv4> ();
      Ipv4InterfaceAddress dest_ipv4_int_addr = destIpv4->GetAddress (1, 0);
      Ipv4Address dest_ip_addr = dest_ipv4_int_addr.GetLocal ();

      InetSocketAddress destAddress (dest_ip_addr, port);
      destAddress.SetTos (0x70); // AC_BE

      uint32_t pktPerSec = pktPerSecBase * intensity;

      if (traffic == "cbr")
        {
          UdpClientHelper source (destAddress);
          source.SetAttribute ("MaxPackets", UintegerValue (pktPerSec * simulationTime));
          source.SetAttribute ("PacketSize", UintegerValue (payloadSize));
          Time interPacketInterval = Seconds (1.0 / pktPerSec);
          source.SetAttribute ("Interval", TimeValue (interPacketInterval)); // packets/s
          ApplicationContainer newSourceApps = source.Install (srcNode);
          sourceApps.Add (newSourceApps);
        }
      else if (traffic == "mmpp")
        {
          // MMPP
          RandomAppHelper source ("ns3::UdpSocketFactory", InetSocketAddress (destAddress));
          source.SetAttribute ("Delay1",
                               StringValue ("ns3::ExponentialRandomVariable[Mean=" +
                                            std::to_string (1.0 / pktPerSec) + "]")); // 0.001
          source.SetAttribute ("Delay2",
                               StringValue ("ns3::ExponentialRandomVariable[Mean=" +
                                            std::to_string (1.0 / pktPerSec * 10) + "]")); // 0.01
          source.SetAttribute ("ModDelay", StringValue ("ns3::ExponentialRandomVariable[Mean=" +
                                                        std::to_string (0.5) + "]"));
          source.SetAttribute ("Size", StringValue ("ns3::ExponentialRandomVariable[Mean=" +
                                                    std::to_string (payloadSize) + "|Bound=2000]"));
          ApplicationContainer newSourceApps = source.Install (srcNode);
          sourceApps.Add (newSourceApps);
        }
      else
        {
          NS_FATAL_ERROR ("traffic must be (cbr|mmpp)");
        }

      if (!sinkNodes.Contains (sinkNode->GetId ()))
        {
          // Create a packet sink to receive these packets
          UdpServerHelper sink (port);
          ApplicationContainer newSinkApps = sink.Install (sinkNode);
          sinkApps.Add (newSinkApps);
          sinkNodes.Add (sinkNode);
        }
    }

  sourceApps.Start (Seconds (0.0));
  sourceApps.Stop (Seconds (simulationTime));
  sinkApps.Start (Seconds (0.0));
  sinkApps.Stop (Seconds (simulationTime));

  // Print node positions
  NS_LOG_UNCOND ("Node Positions:");
  for (NodeContainer::Iterator i = nodes.Begin (); i != nodes.End (); i++)
    {
      Ptr<Node> node = *i;
      Ptr<MobilityModel> mobility = node->GetObject<MobilityModel> ();
      NS_LOG_UNCOND ("---Node ID: " << node->GetId ()
                                    << " Positions: " << mobility->GetPosition ());
    }

  // Configure OpenGym environment
  Ptr<OpenGymInterface> openGymInterface = CreateObject<OpenGymInterface> (openGymPort);
  Ptr<MyGymEnv> myGymEnv = CreateObject<MyGymEnv> (
      srcNodes, Seconds (simulationTime), Seconds (envStepTime), algorithm == "rl", continuous, delayRewardWeight, debug);

  myGymEnv->SetOpenGymInterface (openGymInterface);

  // connect TxOkHeader trace source
  for (uint32_t i = 0; i < nFlows; i++)
    {
      Ptr<Node> node = srcNodes.Get (i);
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
  myGymEnv->PrintResults ();

  openGymInterface->NotifySimulationEnd ();

  Simulator::Destroy ();
}