#include "delay-jitter-estimation-2.h"
#include "ns3/tag.h"
#include "ns3/simulator.h"
#include "ns3/string.h"

namespace ns3 {

class DelayJitterEstimationTimestampTag2 : public Tag
{
public:
  DelayJitterEstimationTimestampTag2 ();

  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;

  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (TagBuffer i) const;
  virtual void Deserialize (TagBuffer i);
  virtual void Print (std::ostream &os) const;

  Time GetTxTime (void) const;

private:
  uint64_t m_creationTime; // The time stored in the tag
};

DelayJitterEstimationTimestampTag2::DelayJitterEstimationTimestampTag2 ()
  : m_creationTime (Simulator::Now ().GetTimeStep ())
{
}

TypeId
DelayJitterEstimationTimestampTag2::GetTypeId (void)
{
  static TypeId tid = TypeId ("anon::DelayJitterEstimationTimestampTag2")
    .SetParent<Tag> ()
    .SetGroupName("Network")
    .AddConstructor<DelayJitterEstimationTimestampTag2> ()
    .AddAttribute ("CreationTime",
                   "The time at which the timestamp was created",
                   StringValue ("0.0s"),
                   MakeTimeAccessor (&DelayJitterEstimationTimestampTag2::GetTxTime),
                   MakeTimeChecker ())
  ;
  return tid;
}

TypeId
DelayJitterEstimationTimestampTag2::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
DelayJitterEstimationTimestampTag2::GetSerializedSize (void) const
{
  return 8;
}

void
DelayJitterEstimationTimestampTag2::Serialize (TagBuffer i) const
{
  i.WriteU64 (m_creationTime);
}

void
DelayJitterEstimationTimestampTag2::Deserialize (TagBuffer i)
{
  m_creationTime = i.ReadU64 ();
}

void
DelayJitterEstimationTimestampTag2::Print (std::ostream &os) const
{
  os << "CreationTime=" << m_creationTime;
}

Time
DelayJitterEstimationTimestampTag2::GetTxTime (void) const
{
  return TimeStep (m_creationTime);
}

DelayJitterEstimation2::DelayJitterEstimation2 ()
  : m_previousRx (Simulator::Now ()),
    m_previousRxTx (Simulator::Now ()),
    m_jitter (0),
    m_delay (Seconds (0.0))
{
}

void
DelayJitterEstimation2::PrepareTx (Ptr<const Packet> packet)
{
  DelayJitterEstimationTimestampTag2 tag;
  packet->AddByteTag (tag);
}

bool
DelayJitterEstimation2::IsMarked (Ptr<const Packet> packet)
{
  DelayJitterEstimationTimestampTag2 tag;
  return packet->FindFirstMatchingByteTag (tag);
}

void
DelayJitterEstimation2::RecordRx (Ptr<const Packet> packet)
{
  DelayJitterEstimationTimestampTag2 tag;
  bool found;
  found = packet->FindFirstMatchingByteTag (tag);
  if (!found)
    {
      return;
    }
  tag.GetTxTime ();

  Time delta = (Simulator::Now () - m_previousRx) - (tag.GetTxTime () - m_previousRxTx);
  m_jitter += (Abs (delta) - m_jitter) / (int64x64_t)16;
  m_previousRx = Simulator::Now ();
  m_previousRxTx = tag.GetTxTime ();
  m_delay = Simulator::Now () - tag.GetTxTime ();
}

Time
DelayJitterEstimation2::GetLastDelay (void) const
{
  return m_delay;
}

uint64_t
DelayJitterEstimation2::GetLastJitter (void) const
{
  return m_jitter.GetHigh ();
}

} // namespace ns3