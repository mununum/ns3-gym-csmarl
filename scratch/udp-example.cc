/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("FirstScriptExample");

int
main (int argc, char *argv[])
{
  CommandLine cmd;
  uint32_t simSeed = 1;
  cmd.AddValue ("simSeed", "Seed for random generator. Default: 1", simSeed);
  cmd.Parse (argc, argv);

  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (simSeed);
  
  Time::SetResolution (Time::NS);
  LogComponentEnable ("RandomGenerator", LOG_LEVEL_INFO);

  NodeContainer nodes;
  nodes.Create (2);

  PointToPointHelper pointToPoint;
  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
  pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));

  NetDeviceContainer devices;
  devices = pointToPoint.Install (nodes);

  InternetStackHelper stack;
  stack.Install (nodes);

  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.0");

  Ipv4InterfaceContainer interfaces = address.Assign (devices);

  uint16_t port = 9;
  double simulationTime = 10.0;

  Ptr<Ipv4> destIpv4 = nodes.Get (1)->GetObject<Ipv4> ();
  Ipv4InterfaceAddress dest_ipv4_int_addr = destIpv4->GetAddress (1, 0);
  Ipv4Address dest_ip_addr = dest_ipv4_int_addr.GetLocal ();
  InetSocketAddress destAddress (dest_ip_addr, port);
  destAddress.SetTos (0x70);

  double delay = 1.0;
  uint32_t size = 1000;
  RandomAppHelper app ("ns3::UdpSocketFactory", InetSocketAddress (destAddress));
  // app.SetAttribute ("Delay", StringValue ("ns3::ExponentialRandomVariable[Mean=1.0]"));
  // app.SetAttribute ("Size", StringValue ("ns3::ExponentialRandomVariable[Mean=1000][Bound=2000]"));
  app.SetAttribute ("Delay", StringValue ("ns3::ExponentialRandomVariable[Mean=" + std::to_string(delay) + "]"));
  app.SetAttribute ("Size", StringValue ("ns3::ExponentialRandomVariable[Mean=" + std::to_string(size) + "][Bound=2000]"));
  app.Install (nodes.Get (0));

  UdpServerHelper sink (port);
  ApplicationContainer sinkApps = sink.Install (nodes.Get (1));

  Simulator::Stop (Seconds (simulationTime));
  Simulator::Run ();
  Simulator::Destroy ();
  return 0;
}
