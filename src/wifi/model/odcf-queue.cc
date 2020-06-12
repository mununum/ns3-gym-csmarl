#include "odcf-queue.h"
#include "ns3/log.h"
#include "ns3/uinteger.h"
#include "ns3/enum.h"
#include "ns3/simulator.h"
#include "ns3/delay-jitter-estimation-2.h"

#undef NS_LOG_APPEND_CONTEXT
#define NS_LOG_APPEND_CONTEXT                              \
  if (Mac48Address ("00:00:00:00:00:00") != GetAddress ()) \
    {                                                      \
      std::clog << "[mac=" << GetAddress () << "] ";       \
    }

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ODcfQueue");

TypeId
ODcfQueue::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ODcfQueue").SetParent<Object> ().AddConstructor<ODcfQueue> ();

  return tid;
}

ODcfQueue::ODcfQueue ()
{
  NS_ASSERT (false);
}

ODcfQueue::ODcfQueue (Mac48Address to, uint32_t mediaAccessQueueMaxPackets,
                      uint32_t controlQueueMaxPackets, Ptr<ODcf> odcf)
    : m_to (to),
      m_lastHoldingDurationInSlot (0),
      m_minMediaAccessQueueSizeForEndingPart (0),
      m_isWaiting (true)
{
  NS_LOG_FUNCTION (this << to << mediaAccessQueueMaxPackets << controlQueueMaxPackets << odcf);

  m_controlQueue = CreateObject<DropTailQueue<Packet>> ();
  m_controlQueue->SetAttribute (
      "MaxSize", QueueSizeValue (QueueSize (QueueSizeUnit::PACKETS, controlQueueMaxPackets)));

  m_mediaAccessQueue = CreateObject<DropTailQueue<Packet>> ();
  m_mediaAccessQueue->SetAttribute (
      "MaxSize", QueueSizeValue (QueueSize (QueueSizeUnit::PACKETS, mediaAccessQueueMaxPackets)));

  m_odcf = odcf;
}

ODcfQueue::~ODcfQueue ()
{
}

void
ODcfQueue::Enqueue (Ptr<Packet> packet)
{
  NS_LOG_INFO (this << " #packets@CQ=" << m_controlQueue->GetNPackets ()
                    << " #packets@MAQ=" << m_mediaAccessQueue->GetNPackets ());
  NS_LOG_FUNCTION (this << packet);

  if (m_controlQueue->Enqueue (packet) == false)
    {
      NS_LOG_LOGIC ("A packet was dropped at CQ");
    }
  else
    {
      NS_LOG_LOGIC ("A packet was inserted at CQ");

      if (m_isWaiting)
        {
          NS_LOG_LOGIC ("MAQ have waited for queueing at CQ");

          m_minMediaAccessQueueSizeForEndingPart = 0;
          m_isWaiting = false;
          EnqueueToMediaAccessQueue ();
        }
    }
}

Ptr<Packet>
ODcfQueue::Dequeue (Mac48Address &to)
{
  // intentionally the order of logs is set
  NS_LOG_INFO (this << " #packets@CQ=" << m_controlQueue->GetNPackets ()
                    << " #packets@MAQ=" << m_mediaAccessQueue->GetNPackets ());
  NS_LOG_FUNCTION (this << to);

  to = m_to;
  return m_mediaAccessQueue->Dequeue ();
}

Ptr<const Packet>
ODcfQueue::Peek ()
{
  // intentionally the order of logs is set
  NS_LOG_INFO (this << " #packets@CQ=" << m_controlQueue->GetNPackets ()
                    << " #packets@MAQ=" << m_mediaAccessQueue->GetNPackets ());
  NS_LOG_FUNCTION (this);

  return m_mediaAccessQueue->Peek ();
}

void
ODcfQueue::EnqueueToMediaAccessQueue ()
{
  NS_LOG_INFO (this << " #packets@CQ=" << m_controlQueue->GetNPackets ()
                    << " #packets@MAQ=" << m_mediaAccessQueue->GetNPackets ());
  NS_LOG_FUNCTION (this);

  Ptr<Packet> packet = m_controlQueue->Dequeue ();
  if (packet == 0)
    {
      NS_LOG_LOGIC ("CQ is empty, so MAQ will be waiting for queueing at CQ");

      m_minMediaAccessQueueSizeForEndingPart = GetMediaAccessQueueSize ();
      m_isWaiting = true;
    }
  else
    {
      DelayJitterEstimation2::PrepareTx (packet, 1);
      m_mediaAccessQueue->Enqueue (packet);
      NS_LOG_LOGIC ("A packet at CQ was inserted at MAQ");

      uint32_t mediaAccessQueueSize = GetMediaAccessQueueSize ();
      m_enqueueToMediaAccessQueueEvent =
          Simulator::Schedule (m_odcf->GetSourceInterval (mediaAccessQueueSize),
                               &ODcfQueue::EnqueueToMediaAccessQueue, this);

      if (mediaAccessQueueSize == 1)
        {
          NS_LOG_LOGIC ("MAQ is not empty from now on");
          m_odcf->NotifyMediaAccessQueueHasPacket (this);
        }
    }
}

void
ODcfQueue::SetLastHoldingDurationInSlot (uint32_t lastHoldingDurationInSlot)
{
  m_lastHoldingDurationInSlot = lastHoldingDurationInSlot;
}

uint32_t
ODcfQueue::GetLastHoldingDurationInSlot ()
{
  return m_lastHoldingDurationInSlot;
}

uint32_t
ODcfQueue::GetMediaAccessQueueSize () const
{
  uint32_t mediaAccessQueueSize = m_mediaAccessQueue->GetNPackets ();
  if (mediaAccessQueueSize == 0)
    {
      return 0;
    }

  uint32_t adjustedMediaAccessQueueSize = mediaAccessQueueSize;

  switch (m_odcf->GetMode ())
    {
    case ODcf::NONE:
      break;
    case ODcf::JUMP_START:
      NS_ASSERT_MSG (false, "Not supported yet");
      break;
    case ODcf::KEEP_END:
      if (m_isWaiting)
        {
          NS_ASSERT (mediaAccessQueueSize <= m_minMediaAccessQueueSizeForEndingPart);

          adjustedMediaAccessQueueSize = m_minMediaAccessQueueSizeForEndingPart;

          NS_LOG_INFO (this << "Use more aggressiveness for an ending part: "
                            << mediaAccessQueueSize << "->" << adjustedMediaAccessQueueSize);
        }
      break;
    case ODcf::JUMP_START_AND_KEEP_END:
      NS_ASSERT_MSG (false, "Not supported yet");
      break;
    default:
      NS_ASSERT (false);
    }

  return adjustedMediaAccessQueueSize;
}

Mac48Address
ODcfQueue::GetTo () const
{
  return m_to;
}

Mac48Address
ODcfQueue::GetAddress () const
{
  if (m_odcf != 0)
    return m_odcf->GetAddress ();

  return Mac48Address ("00:00:00:00:00:00");
}

} // namespace ns3