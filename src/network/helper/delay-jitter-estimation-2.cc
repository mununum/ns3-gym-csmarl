#include "delay-jitter-estimation-2.h"
#include "ns3/tag.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/uinteger.h"

namespace ns3 {

class DelayJitterEstimationTimestampTag2 : public Tag
{
public:
  DelayJitterEstimationTimestampTag2 (uint8_t type);

  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;

  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (TagBuffer i) const;
  virtual void Deserialize (TagBuffer i);
  virtual void Print (std::ostream &os) const;

  Time GetTxTime (void) const;
  uint8_t GetType (void) const;

private:
  uint64_t m_creationTime; // The time stored in the tag
  uint8_t m_type; // type of the tag
};

DelayJitterEstimationTimestampTag2::DelayJitterEstimationTimestampTag2 (uint8_t type = 0)
    : m_creationTime (Simulator::Now ().GetTimeStep ()), m_type (type)
{
}

TypeId
DelayJitterEstimationTimestampTag2::GetTypeId (void)
{
  static TypeId tid =
      TypeId ("anon::DelayJitterEstimationTimestampTag2")
          .SetParent<Tag> ()
          .SetGroupName ("Network")
          .AddConstructor<DelayJitterEstimationTimestampTag2> ()
          .AddAttribute (
              "CreationTime", "The time at which the timestamp was created", StringValue ("0.0s"),
              MakeTimeAccessor (&DelayJitterEstimationTimestampTag2::GetTxTime), MakeTimeChecker ())
          .AddAttribute ("Type", "Integer indicator of tag type", UintegerValue (0),
                         MakeUintegerAccessor (&DelayJitterEstimationTimestampTag2::m_type),
                         MakeUintegerChecker<uint8_t> ());
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
  return 8 + 1; // timestamp + type
}

void
DelayJitterEstimationTimestampTag2::Serialize (TagBuffer i) const
{
  i.WriteU64 (m_creationTime);
  i.WriteU8 (m_type);
}

void
DelayJitterEstimationTimestampTag2::Deserialize (TagBuffer i)
{
  m_creationTime = i.ReadU64 ();
  m_type = i.ReadU8 ();
}

void
DelayJitterEstimationTimestampTag2::Print (std::ostream &os) const
{
  os << "CreationTime=" << m_creationTime;
  os << "Type=" << m_type;
}

Time
DelayJitterEstimationTimestampTag2::GetTxTime (void) const
{
  return TimeStep (m_creationTime);
}

uint8_t
DelayJitterEstimationTimestampTag2::GetType (void) const
{
  return m_type;
}

DelayJitterEstimation2::DelayJitterEstimation2 (uint8_t type)
    : m_previousRx (Simulator::Now ()),
      m_previousRxTx (Simulator::Now ()),
      m_jitter (0),
      m_delay (Seconds (0.0)),
      m_type (type)
{
}

void
DelayJitterEstimation2::PrepareTx (Ptr<const Packet> packet, uint8_t type)
{
  DelayJitterEstimationTimestampTag2 tag (type);
  packet->AddByteTag (tag);
}

bool
DelayJitterEstimation2::FindFirstTypeMatchingByteTag (Ptr<const Packet> packet,
                                                      DelayJitterEstimationTimestampTag2 &tag,
                                                      uint8_t type)
{
  TypeId tid = tag.GetInstanceTypeId ();
  ByteTagIterator i = packet->GetByteTagIterator ();
  DelayJitterEstimationTimestampTag2 temp_tag;
  while (i.HasNext ())
    {
      ByteTagIterator::Item item = i.Next ();
      if (tid == item.GetTypeId ())
        {
          item.GetTag (temp_tag);
          if (temp_tag.GetType () == type) {
            item.GetTag (tag);
            return true;
          }
        }
    }
  return false;
}

bool
DelayJitterEstimation2::IsMarked (Ptr<const Packet> packet, uint8_t type)
{
  DelayJitterEstimationTimestampTag2 tag;
  return FindFirstTypeMatchingByteTag (packet, tag, type);
}

void
DelayJitterEstimation2::RecordRx (Ptr<const Packet> packet)
{
  DelayJitterEstimationTimestampTag2 tag;
  bool found;
  found = FindFirstTypeMatchingByteTag (packet, tag, m_type);
  if (!found)
    {
      return;
    }
  tag.GetTxTime ();

  Time delta = (Simulator::Now () - m_previousRx) - (tag.GetTxTime () - m_previousRxTx);
  m_jitter += (Abs (delta) - m_jitter) / (int64x64_t) 16;
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