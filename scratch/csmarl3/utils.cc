#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/node-list.h"
#include "utils.h"

namespace ns3 {

Graph::Graph ()
{
  nNodes = 0;
  nEdges = 0;
  nFlows = 0;

  pos.clear ();
  edges.clear ();
  flows.clear ();
}

void
Graph::AddNode (const float a, const float b)
{
  pos.push_back (std::make_pair (a, b));
  nNodes++;
}

void
Graph::AddEdge (const uint32_t a, const uint32_t b)
{
  auto ret = edges.insert (std::make_pair (a, b));
  if (ret.second) // insertion took place
    nEdges++;
}

void
Graph::AddFlow (const uint32_t a, const uint32_t b)
{
  flows.push_back (std::make_pair (a, b));
  nFlows++;
}

Ptr<Graph>
ReadLinkGraph (const std::string topology)
{
  Ptr<Graph> graph = CreateObject<Graph> ();
  std::ifstream graph_file;
  const std::string graph_file_name = "experiments/graphs/" + topology + ".txt";
  uint32_t n;

  graph_file.open (graph_file_name);
  if (graph_file.fail ())
    {
      NS_FATAL_ERROR ("File " << graph_file_name << " not found");
    }

  // position
  graph_file >> n;
  for (uint32_t i = 0; i < n; i++)
    {
      float a, b;
      graph_file >> a >> b;

      // sender and receiver in identical locations
      graph->AddNode (a, b);
      graph->AddNode (a, b);

      // establish a link between tx and rx
      uint32_t tx = i * 2;
      uint32_t rx = i * 2 + 1;
      graph->AddEdge (tx, rx);
      graph->AddFlow (tx, rx);
    }

  // interferences between links
  graph_file >> n;
  for (uint32_t i = 0; i < n; i++)
    {
      uint32_t a, b;
      graph_file >> a >> b;

      uint32_t a_tx = a * 2;
      uint32_t a_rx = a * 2 + 1;
      uint32_t b_tx = b * 2;
      uint32_t b_rx = b * 2 + 1;

      graph->AddEdge (a_tx, b_tx);
      graph->AddEdge (a_tx, b_rx);
      graph->AddEdge (a_rx, b_tx);
      graph->AddEdge (a_rx, b_rx);
    }

  graph_file.close ();

  return graph;
}

void
ConfigureMatrixTopology (const Ptr<MultiModelSpectrumChannel> spectrumChannel, 
                         const Ptr<const Graph> graph, NodeContainer &nodes)
{
  Ptr<MatrixPropagationLossModel> lossModel = CreateObject<MatrixPropagationLossModel> ();

  for (auto it = graph->EdgeBegin (); it != graph->EdgeEnd (); it++)
    {
      Ptr<MobilityModel> mobilityA = nodes.Get (it->first)->GetObject<MobilityModel> ();
      Ptr<MobilityModel> mobilityB = nodes.Get (it->second)->GetObject<MobilityModel> ();

      lossModel->SetLoss (mobilityA, mobilityB, 0);  // symmetric
    }

  spectrumChannel->AddPropagationLossModel (lossModel);
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

} // namespace ns3