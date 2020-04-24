#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/config-store-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "ns3/propagation-loss-model.h"

using namespace ns3;

static void PopulateArpCache ();

static void ConfigureFlowInTheMiddleTopology (Ptr<YansWifiChannel> channel, NodeContainer &nodes);
static void PacketSinkCallback (uint32_t *cumulativeBytes, Ptr<const Packet> packet,
                                const Address &address);

int
main (int argc, char *argv[])
{
  // Configuration
  Time simulationEnd = Seconds (100.0);
  // ODcf parameters
  std::string macType = "ns3::AdhocWifiMac";
  // std::string macType = "ns3::ODcfAdhocWifiMac";
  Config::SetDefault ("ns3::ODcf::V", DoubleValue (400.0));
  Config::SetDefault ("ns3::ODcf::CQ_max", UintegerValue (100));
  Config::SetDefault ("ns3::ODcf::Q_max", UintegerValue (1000));

  // NIC parameters
  // Config::SetDefault ("ns3::WifiMacQueue::MaxPacketNumber", UintegerValue (1)); // MYTODO deprecated
  // Config::SetDefault ("ns3::WifiMacQueue::MaxDelay", TimeValue (simulationEnd));
  std::string phyRate ("6Mbps");
  // traffic parameters
  uint32_t packetSize = 1000; // bytes
  uint32_t nFlows = 3;

  // LogComponentEnable ("ODcf", LOG_LEVEL_DEBUG);
  // LogComponentEnable ("ODcfTxop", LOG_LEVEL_DEBUG);
  // LogComponentEnable ("ODcfAdhocWifiMac", LOG_LEVEL_DEBUG);
  // LogComponentEnable ("MacLow", LOG_LEVEL_DEBUG);
  // LogComponentEnable ("Txop", LOG_LEVEL_DEBUG);
  // LogComponentEnable ("WifiPhy", LOG_LEVEL_DEBUG);

  // Nodes
  NodeContainer nodes;
  nodes.Create (nFlows * 2);

  // WiFi
  WifiHelper wifi;
  wifi.SetStandard (WIFI_PHY_STANDARD_80211a);
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager", "DataMode",
                                StringValue ("OfdmRate" + phyRate), "ControlMode",
                                StringValue ("OfdmRate6Mbps")); // MYTODO check compatibility
  // Propagation model: graph based
  YansWifiChannelHelper wifiChannel;
  wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
  wifiChannel.AddPropagationLoss ("ns3::MatrixPropagationLossModel");
  Ptr<YansWifiChannel> wifiChannelInstance = wifiChannel.Create ();
  // Phy
  YansWifiPhyHelper wifiPhy = YansWifiPhyHelper::Default ();
  wifiPhy.Set ("RxGain", DoubleValue (0));
  wifiPhy.SetChannel (wifiChannelInstance);
  // Non-QoS MAC
  WifiMacHelper wifiMac;
  wifiMac.SetType (macType, "QosSupported", BooleanValue (false));
  // Install Wifi capabilities
  NetDeviceContainer devices = wifi.Install (wifiPhy, wifiMac, nodes);

  // Location - mo meaning since we use a graph-based link connectivity model
  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = Create<ListPositionAllocator> ();
  for (uint32_t i = 0; i < nFlows; i++)
    {
      positionAlloc->Add (Vector (0.0, 0.0, 0.0));
      positionAlloc->Add (Vector (5.0, 0.0, 0.0));
    }
  mobility.SetPositionAllocator (positionAlloc);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (nodes);

  // config graph-based model
  ConfigureFlowInTheMiddleTopology (wifiChannelInstance, nodes);

  // Internet protocols
  InternetStackHelper internet;
  internet.Install (nodes);

  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces = ipv4.Assign (devices);

  uint32_t *cumulativeBytes = new uint32_t[nFlows];
  // Applications
  for (uint32_t i = 0; i < nFlows; i++)
    {
      OnOffHelper onOffSource ("ns3::UdpSocketFactory",
                               InetSocketAddress (interfaces.GetAddress (i + nFlows), 80));
      onOffSource.SetAttribute ("MaxBytes", UintegerValue (0)); // infinite backlog
      onOffSource.SetAttribute ("PacketSize", UintegerValue (packetSize));
      onOffSource.SetAttribute ("OnTime",
                                StringValue ("ns3::ConstantRandomVariable[Constant=" +
                                             std::to_string (simulationEnd.GetSeconds ()) + "]"));
      onOffSource.SetAttribute ("OffTime",
                                StringValue ("ns3::ConstantRandomVariable[Constant=0.0]"));
      onOffSource.SetAttribute ("DataRate", DataRateValue (DataRate (phyRate))); // saturated

      ApplicationContainer sourceApp = onOffSource.Install (nodes.Get (i));
      sourceApp.Start (Seconds (0.0));
      sourceApp.Stop (simulationEnd);

      PacketSinkHelper sink ("ns3::UdpSocketFactory",
                             InetSocketAddress (Ipv4Address::GetAny (), 80));
      ApplicationContainer sinkApp = sink.Install (nodes.Get (i + nFlows));
      sinkApp.Start (Seconds (0.0));
      sinkApp.Stop (simulationEnd);

      cumulativeBytes[i] = 0;
      sinkApp.Get (0)->GetObject<PacketSink> ()->TraceConnectWithoutContext (
          "Rx", MakeBoundCallback (PacketSinkCallback, &cumulativeBytes[i]));
    }

  PopulateArpCache (); // MYTODO check compatibility

  Simulator::Stop (simulationEnd);

  Simulator::Run ();

  for (uint32_t i = 0; i < nFlows; i++)
    {
      std::cout << "Flow " << i << ": "
                << cumulativeBytes[i] * 8.0 / simulationEnd.GetSeconds () / 1000000 << " Mb/s"
                << std::endl;
    }

  delete[] cumulativeBytes;

  Simulator::Destroy ();

  return 0;
}

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

void
ConfigureFlowInTheMiddleTopology (Ptr<YansWifiChannel> channel, NodeContainer &nodes)
{
  PointerValue tmp;
  channel->GetAttribute ("PropagationLossModel", tmp);
  Ptr<MatrixPropagationLossModel> lossModel =
      tmp.GetObject ()->GetObject<MatrixPropagationLossModel> ();

  // middle flow
  Ptr<MobilityModel> mobilityR = nodes.Get (0)->GetObject<MobilityModel> ();
  Ptr<MobilityModel> mobilityS = nodes.Get (nodes.GetN () / 2)->GetObject<MobilityModel> ();
  for (uint32_t i = 1; i < nodes.GetN (); i++)
    {
      if (i != nodes.GetN () / 2)
        {
          Ptr<MobilityModel> mobility = nodes.Get (i)->GetObject<MobilityModel> ();

          lossModel->SetLoss (mobilityR, mobility, 0); // symmetric
          lossModel->SetLoss (mobilityS, mobility, 0); // symmetric
        }
    }

  // transmitter-receiver pair
  for (uint32_t i = 0; i < nodes.GetN () / 2; i++)
    {
      mobilityR = nodes.Get (i)->GetObject<MobilityModel> ();
      mobilityS = nodes.Get (i + nodes.GetN () / 2)->GetObject<MobilityModel> ();

      lossModel->SetLoss (mobilityR, mobilityS, 0); // symmetric
    }
}

void
PacketSinkCallback (uint32_t *cumulativeBytes, Ptr<const Packet> packet, const Address &address)
{
  *cumulativeBytes += packet->GetSize ();
}