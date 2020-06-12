#ifndef DELAY_JITTER_ESTIMATION_2_H
#define DELAY_JITTER_ESTIMATION_2_H

#include "ns3/object.h"
#include "ns3/nstime.h"
#include "ns3/packet.h"

namespace ns3 {

class DelayJitterEstimationTimestampTag2;

// A copy of DelayJitterEstimation class for calculating multiple levels of delay
class DelayJitterEstimation2 : public Object
{
public:
  DelayJitterEstimation2 (uint8_t type = 0);

  static void PrepareTx (Ptr<const Packet> packet, uint8_t type = 0);

  static bool IsMarked (Ptr<const Packet> packet, uint8_t type = 0);

  void RecordRx (Ptr<const Packet> packet);

  Time GetLastDelay (void) const;

  uint64_t GetLastJitter (void) const;

private:
  static bool FindFirstTypeMatchingByteTag (Ptr<const Packet> packet,
                                            DelayJitterEstimationTimestampTag2 &tag, uint8_t type);

  Time m_previousRx; // Previous Rx time
  Time m_previousRxTx; // Previous Rx or Tx time
  int64x64_t m_jitter; // Jitter estimation
  Time m_delay; // Delay estimation
  uint8_t m_type; // type of this instance
};

} // namespace ns3

#endif /* DELAY_JITTER_ESTIMATION_2_H */