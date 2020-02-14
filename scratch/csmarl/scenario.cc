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

} // namespace ns3