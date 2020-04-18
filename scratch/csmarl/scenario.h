#ifndef SCENARIO_H
#define SCENARIO_H

#include "ns3/spectrum-module.h"

namespace ns3 {

void ConfigureFCTopology (Ptr<MultiModelSpectrumChannel> spectrumChannel, NodeContainer &nodes);
void ConfigureFIMTopology (Ptr<MultiModelSpectrumChannel> spectrumChannel, NodeContainer &nodes);
void ConfigureMatrixTopology (Ptr<MultiModelSpectrumChannel> spectrumChannel, NodeContainer &nodes,
                              uint32_t nEdges, std::vector<std::tuple<uint32_t, uint32_t>> &edges);
void ReadGraph (std::string topology, uint32_t &nNodes, uint32_t &nEdges, uint32_t &nFlows,
                std::vector<std::tuple<float, float>> &pos,
                std::vector<std::tuple<uint32_t, uint32_t>> &edges,
                std::vector<std::tuple<uint32_t, uint32_t>> &flows);
void MakeFlows (uint32_t nNodes, uint32_t nEdges, uint32_t nFlows,
                std::vector<std::tuple<uint32_t, uint32_t>> edges,
                std::vector<std::tuple<uint32_t, uint32_t>> &flows,
                uint32_t seed);

} // namespace ns3

#endif // SCENARIO_H