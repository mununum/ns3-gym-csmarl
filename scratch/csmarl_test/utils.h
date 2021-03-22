#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <utility>
#include "ns3/spectrum-module.h"

namespace ns3 {

class MyConfig
{
public:
  uint32_t openGymPort = 5555;
  uint32_t simSeed = 0;
  uint32_t graphSeed = 0;
  double simTime = 20;
  double stepTime = 0.005;

  std::string layout = "node";  // layout: (node|link)
  std::string loss = "graph";  // loss: (graph|geometric)

  // options used in loss==graph
  std::string topology = "single";

  // options used in loss==geometric
  uint32_t nFlows = 3;
  double threshold = 0.5;

  std::string mobility = "fixed";  // mobility: (fixed|paired|random)
  double intensity = 1.0;

  std::string algorithm = "80211";
  bool debug = false;
};

// undirected graph structure
class Graph : public Object
{
public:
  Graph ();

  void AddNode (const float, const float);
  void AddEdge (const uint32_t, const uint32_t);
  void AddFlow (const uint32_t, const uint32_t);

  uint32_t GetNNodes () const { return nNodes; }
  uint32_t GetNEdges () const { return nEdges; }
  uint32_t GetNFlows () const { return nFlows; }

public:
  std::vector<std::pair<float, float>>::const_iterator
  PosBegin () const { return pos.begin (); }

  std::vector<std::pair<float, float>>::const_iterator
  PosEnd () const { return pos.end (); }

  std::set<std::pair<uint32_t, uint32_t>>::const_iterator
  EdgeBegin () const { return edges.begin (); }

  std::set<std::pair<uint32_t, uint32_t>>::const_iterator
  EdgeEnd () const { return edges.end (); }

  std::vector<std::pair<uint32_t, uint32_t>>::const_iterator
  FlowBegin () const { return flows.begin (); }

  std::vector<std::pair<uint32_t, uint32_t>>::const_iterator
  FlowEnd () const { return flows.end (); }

private:
  uint32_t nNodes;
  uint32_t nEdges;
  uint32_t nFlows;

  std::vector<std::pair<float, float>> pos;
  std::set<std::pair<uint32_t, uint32_t>> edges;
  std::vector<std::pair<uint32_t, uint32_t>> flows;
};

Ptr<Graph> BuildGraph (const MyConfig &config);

void ConfigureMatrixTopology (const Ptr<MultiModelSpectrumChannel>,
                              const Ptr<const Graph>, NodeContainer &);

void PopulateArpCache ();

} // namespace ns3

#endif // UTILS_H