#include <random>
#include <algorithm>
#include "scenario.h"

namespace ns3 {

void
ConfigureFCTopology (Ptr<MultiModelSpectrumChannel> spectrumChannel, NodeContainer &nodes)
{
  uint32_t nodeNum = nodes.GetN ();
  Ptr<MatrixPropagationLossModel> lossModel = CreateObject<MatrixPropagationLossModel> ();

  // Configure FC topology

  for (uint32_t i = 0; i < nodeNum; i++)
    {
      Ptr<MobilityModel> mobilityA = nodes.Get (i)->GetObject<MobilityModel> ();
      for (uint32_t j = i + 1; j < nodeNum; j++)
        {
          Ptr<MobilityModel> mobilityB = nodes.Get (j)->GetObject<MobilityModel> ();

          lossModel->SetLoss (mobilityA, mobilityB, 0);
        }
    }

  spectrumChannel->AddPropagationLossModel (lossModel);
}

void
ConfigureFIMTopology (Ptr<MultiModelSpectrumChannel> spectrumChannel, NodeContainer &nodes)
{
  uint32_t nodeNum = nodes.GetN ();
  Ptr<MatrixPropagationLossModel> lossModel = CreateObject<MatrixPropagationLossModel> ();

  // Configure FIM topology
  // Example: For nodes 0, 1, 2, 3, 4, 5,
  // 0 -- 1, 2 -- 3, 4 -- 5 will be pairs
  // 0 -- 1 will be middle flow

  Ptr<MobilityModel> mobilityS = nodes.Get (0)->GetObject<MobilityModel> ();
  Ptr<MobilityModel> mobilityR = nodes.Get (1)->GetObject<MobilityModel> ();

  // Configure inter-flow interference
  for (uint32_t i = 2; i < nodeNum; i++)
    {
      Ptr<MobilityModel> mobility = nodes.Get (i)->GetObject<MobilityModel> ();

      lossModel->SetLoss (mobilityS, mobility, 0);
      lossModel->SetLoss (mobilityR, mobility, 0);
    }

  // Configure intra-flow interference
  for (uint32_t srcNodeId = 0; srcNodeId < nodeNum; srcNodeId += 2)
    {
      uint32_t dstNodeId = srcNodeId + 1;
      mobilityS = nodes.Get (srcNodeId)->GetObject<MobilityModel> ();
      mobilityR = nodes.Get (dstNodeId)->GetObject<MobilityModel> ();

      lossModel->SetLoss (mobilityS, mobilityR, 0);
    }

  spectrumChannel->AddPropagationLossModel (lossModel);
}

void
ConfigureMatrixTopology (Ptr<MultiModelSpectrumChannel> spectrumChannel, NodeContainer &nodes,
                         uint32_t nEdges, std::vector<std::tuple<uint32_t, uint32_t>> &edges)
{
  Ptr<MatrixPropagationLossModel> lossModel = CreateObject<MatrixPropagationLossModel> ();

  for (uint32_t i = 0; i < nEdges; i++)
    {
      uint32_t a = std::get<0> (edges[i]);
      uint32_t b = std::get<1> (edges[i]);

      Ptr<MobilityModel> mobilityA = nodes.Get (a)->GetObject<MobilityModel> ();
      Ptr<MobilityModel> mobilityB = nodes.Get (b)->GetObject<MobilityModel> ();

      lossModel->SetLoss (mobilityA, mobilityB, 0);
    }

  spectrumChannel->AddPropagationLossModel (lossModel);
}

void
ReadGraph (std::string topology, uint32_t &nNodes, uint32_t &nEdges, uint32_t &nFlows,
           std::vector<std::tuple<float, float>> &pos,
           std::vector<std::tuple<uint32_t, uint32_t>> &edges,
           std::vector<std::tuple<uint32_t, uint32_t>> &flows)
{
  std::ifstream graph_file;
  std::string graph_file_name = "scratch/csmarl/graphs/" + topology + ".txt";

  graph_file.open (graph_file_name);
  if (graph_file.fail ())
    {
      NS_FATAL_ERROR ("File " << graph_file_name << " not found");
    }
  graph_file >> nNodes;
  for (uint32_t i = 0; i < nNodes; i++)
    {
      float a, b;
      graph_file >> a >> b;
      pos.push_back (std::make_tuple (a, b));
    }
  graph_file >> nEdges;
  for (uint32_t i = 0; i < nEdges; i++)
    {
      uint32_t a, b;
      graph_file >> a >> b;
      edges.push_back (std::make_tuple (a, b));
    }
  graph_file >> nFlows;
  for (uint32_t i = 0; i < nFlows; i++)
    {
      uint32_t a, b;
      graph_file >> a >> b;
      flows.push_back (std::make_tuple (a, b));
    }
  graph_file.close ();
}

void
MakeFlows (uint32_t nNodes, uint32_t nEdges, uint32_t nFlows,
           std::vector<std::tuple<uint32_t, uint32_t>> edges,
           std::vector<std::tuple<uint32_t, uint32_t>> &flows, uint32_t seed)
{

  std::map<uint32_t, std::vector<uint32_t>> neighbors;
  std::vector<uint32_t> nodes;

  for (uint32_t i = 0; i < nEdges; i++)
    {
      uint32_t a, b;
      a = std::get<0> (edges[i]);
      b = std::get<1> (edges[i]);
      neighbors[a].push_back (b);
      neighbors[b].push_back (a);
    }

  for (uint32_t i = 0; i < nNodes; i++)
    {
      nodes.push_back (i);
      // std::cout << i << ": ";
      // for (std::vector<uint32_t>::iterator j = neighbors[i].begin (); j != neighbors[i].end (); j++)
      //   {
      //     std::cout << (*j) << " ";
      //   }
      // std::cout << std::endl;
    }

  std::random_device rd;
  std::mt19937 g (rd ());
  g.seed (seed);
  std::shuffle (nodes.begin (), nodes.end (), g);

  // for (uint32_t i = 0; i < nFlows; i++)
  //   {
  //     std::cout << std::get<0> (flows[i]) << " " << std::get<1> (flows[i]) << std::endl;
  //   }

  // flows will be overwritten
  flows.clear ();

  // randomly choose the flows
  // std::cout << "MakeFlows" << std::endl;

  for (uint32_t i = 0; i < nFlows; i++)
    {
      uint32_t a = nodes[i];

      uint32_t n = std::rand () % neighbors[a].size ();
      uint32_t b = neighbors[a][n];

      // make a->b flow
      flows.push_back (std::make_tuple (a, b));
    }
  
  // for (uint32_t i = 0; i < nFlows; i++)
  //   {
  //     std::cout << std::get<0> (flows[i]) << " " << std::get<1> (flows[i]) << std::endl;
  //   }


  // std::exit (EXIT_SUCCESS);
}

} // namespace ns3