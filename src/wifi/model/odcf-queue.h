#ifndef ODCF_QUEUE_H
#define ODCF_QUEUE_H

#include "ns3/object.h"
#include "ns3/packet.h"
#include "ns3/pointer.h"
#include "ns3/nstime.h"
#include "ns3/event-id.h"
#include "ns3/mac48-address.h"
#include "ns3/drop-tail-queue.h"
#include "odcf.h"

namespace ns3 {

class ODcf;

class ODcfQueue : public Object
{
public:
  static TypeId GetTypeId (void);

  ODcfQueue ();
  ODcfQueue (Mac48Address to, uint32_t mediaAccessQueueMaxPackets, uint32_t controlQueueMaxPackets,
             Ptr<ODcf> odcf);
  virtual ~ODcfQueue ();

  void Enqueue (Ptr<Packet> packet);
  Ptr<Packet> Dequeue (Mac48Address &to);
  Ptr<const Packet> Peek ();

  void SetLastHoldingDurationInSlot (uint32_t lastHoldingDurationInSlot);
  uint32_t GetLastHoldingDurationInSlot ();
  uint32_t GetMediaAccessQueueSize () const;
  Mac48Address GetTo () const;
  Mac48Address GetAddress () const; // the address to which this queue is attached

private:
  void EnqueueToMediaAccessQueue ();

  Ptr<ODcf> m_odcf;

  Ptr<DropTailQueue<Packet>> m_controlQueue;
  Ptr<DropTailQueue<Packet>> m_mediaAccessQueue;
  Mac48Address m_to;
  uint32_t m_lastHoldingDurationInSlot; // # of slots, deficit counter
  uint32_t m_minMediaAccessQueueSizeForEndingPart;

  EventId m_enqueueToMediaAccessQueueEvent;
  bool m_isWaiting;
};

} // namespace ns3

#endif // ODCF_QUEUE_H