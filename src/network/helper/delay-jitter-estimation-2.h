#ifndef DELAY_JITTER_ESTIMATION_2_H
#define DELAY_JITTER_ESTIMATION_2_H

#include "ns3/object.h"
#include "ns3/nstime.h"
#include "ns3/packet.h"

namespace ns3 {

// A copy of DelayJitterEstimation class for calculating multiple levels of delay
class DelayJitterEstimation2 : public Object
{
public:
  DelayJitterEstimation2 ();

  static void PrepareTx (Ptr<const Packet> packet);

  static bool IsMarked (Ptr<const Packet> packet);

  void RecordRx (Ptr<const Packet> packet);

  Time GetLastDelay (void) const;

  uint64_t GetLastJitter (void) const;

private:
  Time m_previousRx; // Previous Rx time
  Time m_previousRxTx; // Previous Rx or Tx time
  int64x64_t m_jitter; // Jitter estimation
  Time m_delay; // Delay estimation
};

} // namespace ns3

#endif /* DELAY_JITTER_ESTIMATION_2_H */