#ifndef SCENARIO_H
#define SCENARIO_H

#include "ns3/spectrum-module.h"

namespace ns3 {

void ConfigureFCTopology (Ptr<MultiModelSpectrumChannel> spectrumChannel, NodeContainer &nodes);
void ConfigureFIMTopology (Ptr<MultiModelSpectrumChannel> spectrumChannel, NodeContainer &nodes);

} // namespace ns3

#endif // SCENARIO_H