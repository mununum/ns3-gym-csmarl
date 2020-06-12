#ifndef ODCF_H
#define ODCF_H

#include "ns3/object.h"
#include "ns3/packet.h"
#include "ns3/pointer.h"
#include "ns3/mac48-address.h"
#include "odcf-queue.h"
#include "ns3/nstime.h"
#include "ns3/event-id.h"
#include "ns3/wifi-mac.h"
#include "odcf-txop.h"
#include "odcf-adhoc-wifi-mac.h"
#include <list>
#include <limits.h>

namespace ns3 {

class ODcfTxop;
class ODcfAdhocWifiMac;
class ODcfQueue;

// Transmission intensity, holding duration, link scheduling
class ODcf : public Object {

  friend class ODcfAdhocWifiMac;

public:
  enum Mode {NONE, JUMP_START, KEEP_END, JUMP_START_AND_KEEP_END};
  enum Variant {O_DCF, CW_ADAPTATION, MU_ADAPTATION, DIFF_Q};

  static TypeId GetTypeId (void);

  ODcf ();
  ODcf (Ptr<ODcfAdhocWifiMac> mac, Ptr<ODcfTxop> txop, uint32_t maxCw);
  virtual ~ODcf ();

  void Enqueue (Ptr<Packet> packet, const Mac48Address& to);

  void NotifyMediaAccessQueueHasPacket (Ptr<ODcfQueue> queue);
  void NotifyTransmissionSuccess (uint32_t cwSucc = 0);
  void NotifyTransmissionFailure ();
  void NotifyCurrentMsduTransmissionDurationInSlot (uint32_t nSlots);

  Ptr<const Packet> PeekNextPacket ();
  Time GetSourceInterval (uint32_t mediaAccessQueueSize) const;
  Mode GetMode () const;

  Mac48Address GetAddress () const;

  void SetCw (uint32_t minCw);  // called by RL agent
  uint32_t GetMAQLength ();

private:
  Ptr<ODcfQueue> Find (const Mac48Address& to) const;
  Ptr<ODcfQueue> LinkSchedule ();
  void SendFirstPacketInThisBurst (Ptr<Packet> packet, const Mac48Address to);
  bool SendSubsequentPacketInThisBurstIfPossible ();
  void SendSubsequentPacketInThisBurst (Ptr<Packet> packet, const Mac48Address to);
  double GetTransmissionAggressiveness (uint32_t mediaAccessQueueSize) const;
  uint32_t GetMinCw ();
  void UpdateHoldingDuration (uint32_t succCw = 0);
  bool WillBeFirstImmediateAccess ();
  void FinalizeBurst ();

  Ptr<ODcfAdhocWifiMac> m_mac;
  Ptr<ODcfTxop> m_txop;
  uint32_t m_maxCw;

  Ptr<ODcfQueue> m_currentQueue;
  double m_currentTransmissionIntensity;
  uint32_t m_currentHoldingDurationInSlot; // # of slots
  static const uint32_t MAX_HOLDING_DURATION_IN_SLOT = 10000;
  static const uint32_t A_PACKET_DURATION_IN_SLOT = 150;

  uint32_t m_formerMsduTransmissionDurationInSlot; // # of slots

  bool m_isHolding;

  typedef std::list<Ptr<ODcfQueue> > ODcfQueues;
  typedef ODcfQueues::const_iterator Iterator;
  ODcfQueues m_queues;

  uint32_t m_minQ;
  uint32_t m_maxQ;
  double m_V;
  double m_b;
  uint32_t m_controlQueueMaxPackets;
  double m_C;
  Mode m_mode;

  bool m_RLmode;
  uint32_t m_minCw;  // only used in RL mode

  Variant m_variant;
};

}

#endif // ODCF_H