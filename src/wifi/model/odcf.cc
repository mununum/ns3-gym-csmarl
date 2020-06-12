#include "odcf.h"
#include "odcf-queue.h"
#include "ns3/log.h"
#include "ns3/uinteger.h"
#include "ns3/enum.h"
#include "ns3/simulator.h"
#include "ns3/double.h"

#undef NS_LOG_APPEND_CONTEXT
#define NS_LOG_APPEND_CONTEXT                              \
  if (Mac48Address ("00:00:00:00:00:00") != GetAddress ()) \
    {                                                      \
      std::clog << "[mac=" << GetAddress () << "] ";       \
    }

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ODcf");

NS_OBJECT_ENSURE_REGISTERED (ODcf);

TypeId
ODcf::GetTypeId (void)
{
  static TypeId tid =
      TypeId ("ns3::ODcf")
          .SetParent<Object> ()
          .AddConstructor<ODcf> ()
          .AddAttribute (
              "Q_min", // packets
              "The minimum size of Media Access Queue to prevent too severe oscillations of Q^M",
              UintegerValue (1), MakeUintegerAccessor (&ODcf::m_minQ),
              MakeUintegerChecker<uint32_t> (1))
          .AddAttribute (
              "Q_max", //packets
              "The maximum size of Media Access Queue (the physical limit of Media Access Queue)",
              UintegerValue (1000), MakeUintegerAccessor (&ODcf::m_maxQ),
              MakeUintegerChecker<uint32_t> (1))
          .AddAttribute ("CQ_max", // packets
                         "The physical limit of Control Queue", UintegerValue (100),
                         MakeUintegerAccessor (&ODcf::m_controlQueueMaxPackets),
                         MakeUintegerChecker<uint32_t> (1))
          .AddAttribute ("V", "The accuracy of Utility-Optimal algorithm", DoubleValue (500.0),
                         MakeDoubleAccessor (&ODcf::m_V), MakeDoubleChecker<double> (0.0))
          .AddAttribute ("b", "The step size", DoubleValue (0.01), MakeDoubleAccessor (&ODcf::m_b),
                         MakeDoubleChecker<double> (0.0))
          .AddAttribute ("C", "This controls the inflection point in the sigmoid function",
                         DoubleValue (500), MakeDoubleAccessor (&ODcf::m_C),
                         MakeDoubleChecker<double> (0.0))
          .AddAttribute ("Mode",
                         "Set the heuristics of transmission aggressiveness for starting and "
                         "ending parts of a flow",
                         EnumValue (NONE), MakeEnumAccessor (&ODcf::m_mode),
                         MakeEnumChecker (NONE, "_", JUMP_START, "_js", KEEP_END, "_ke",
                                          JUMP_START_AND_KEEP_END, "_js_ke"))
          .AddAttribute ("RL_mode",
                         "Indicator of whether this odcf module is in RL mode",
                         BooleanValue (false), 
                         MakeBooleanAccessor (&ODcf::m_RLmode),
                         MakeBooleanChecker ());

  return tid;
}

ODcf::ODcf ()
{
  NS_LOG_FUNCTION_NOARGS ();

  NS_ASSERT (false);
}

ODcf::ODcf (Ptr<ODcfAdhocWifiMac> mac, Ptr<ODcfTxop> txop, uint32_t maxCw)
    : m_mac (mac),
      m_txop (txop),
      m_maxCw (maxCw),
      m_currentQueue (0),
      m_currentTransmissionIntensity (0),
      m_currentHoldingDurationInSlot (0),
      m_isHolding (false)
{
  NS_LOG_FUNCTION (this << mac << txop << maxCw);

  m_txop->SetODcf (this);
}

ODcf::~ODcf ()
{
  NS_LOG_FUNCTION_NOARGS ();
}

void
ODcf::Enqueue (Ptr<Packet> packet, const Mac48Address &to)
{
  NS_LOG_FUNCTION (this << packet << to);

  Ptr<ODcfQueue> queue = Find (to);
  if (queue != 0)
    {
      queue->Enqueue (packet);
    }
  else
    {
      NS_LOG_LOGIC ("Queue not created yet for " << to);

      NS_ASSERT (m_minQ < m_maxQ);

      // to, maxQ of MAQ, maxQ of CQ, odcf
      queue = CreateObject<ODcfQueue> (to, m_maxQ, m_controlQueueMaxPackets, this);
      m_queues.push_front (queue);
      queue->Enqueue (packet);
    }
}

Ptr<ODcfQueue>
ODcf::Find (const Mac48Address &to) const
{
  NS_LOG_FUNCTION (this << to);

  for (Iterator iterator = m_queues.begin (); iterator != m_queues.end (); iterator++)
    {
      Ptr<ODcfQueue> queue = *iterator;
      if (queue->GetTo () == to)
        {
          return queue;
        }
    }

  return 0;
}

void
ODcf::SetCw (uint32_t minCw)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (m_RLmode == true);

  m_minCw = minCw;
  m_maxCw = minCw;  // disable BEB
}

uint32_t
ODcf::GetMAQLength ()
{
  NS_LOG_FUNCTION (this);

  if (m_currentQueue == 0)
    return 0;

  return m_currentQueue->GetMediaAccessQueueSize ();
}

uint32_t
ODcf::GetMinCw ()
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (m_currentQueue != 0);

  // if operating in RL mode, then just return current CW
  if (m_RLmode)
    return m_minCw;

  double transmissionAggressiveness =
      GetTransmissionAggressiveness (m_currentQueue->GetMediaAccessQueueSize ());
  m_currentTransmissionIntensity = exp (transmissionAggressiveness);
  double rawCw = 2.0 * (m_currentTransmissionIntensity + m_C) / m_currentTransmissionIntensity;

  uint32_t cw = 1;
  while (cw - rawCw <= 0)
    {
      cw <<= 1;
    }
  if (cw - rawCw > cw >> 2)
    {
      cw >>= 1;
    }
  NS_ASSERT (cw > 1);

  NS_LOG_INFO (Simulator::Now ().GetSeconds ()
               << " " << m_currentQueue->GetMediaAccessQueueSize ()
               << " TA=" << transmissionAggressiveness << " TI=" << m_currentTransmissionIntensity
               << " rawCw=" << rawCw << " cw=" << cw);

  return cw - 1;
}

bool
ODcf::WillBeFirstImmediateAccess ()
{
  return m_txop->GetAifsn () != 0;
}

void
ODcf::UpdateHoldingDuration (uint32_t succCw)
{
  NS_LOG_FUNCTION (this << succCw);

  if (m_RLmode)
    {
      m_currentHoldingDurationInSlot = 0;
      return;
    }

  if (WillBeFirstImmediateAccess ())
    {
      NS_LOG_LOGIC ("Next transmission is immediately accessed");
      NS_ASSERT (succCw > 0);

      // calculate the holding time from succCw
      double accessProbability = std::min (2.0 / succCw, 1.0);
      m_currentHoldingDurationInSlot =
          std::min (m_currentQueue->GetLastHoldingDurationInSlot () +
                        uint32_t (m_currentTransmissionIntensity / accessProbability + 0.5),
                    uint32_t (MAX_HOLDING_DURATION_IN_SLOT));

      NS_LOG_INFO ("accessProbability=" << accessProbability << " holdingDurationInSlot="
                                        << m_currentHoldingDurationInSlot);
    }

  NS_ASSERT (m_currentHoldingDurationInSlot <= MAX_HOLDING_DURATION_IN_SLOT);

  // subtract the former transmission
  if (m_currentHoldingDurationInSlot > m_formerMsduTransmissionDurationInSlot)
    {
      m_currentHoldingDurationInSlot -= m_formerMsduTransmissionDurationInSlot;
    }
  else
    {
      // can occur
      // the reverse case of deficit counter
      NS_LOG_LOGIC ("The former transmission is longer than a given holding duration");
      m_currentHoldingDurationInSlot = 0;
    }

  NS_LOG_INFO ("holdingDurationInSlot=" << m_currentHoldingDurationInSlot
                                        << " m_formerMsduTransmissionDurationInSlot="
                                        << m_formerMsduTransmissionDurationInSlot);
}

void
ODcf::FinalizeBurst ()
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (m_currentQueue != 0);

  m_isHolding = false;
  // deficit counter
  m_currentQueue->SetLastHoldingDurationInSlot (m_currentHoldingDurationInSlot);
  m_currentHoldingDurationInSlot = 0;
  m_currentQueue = 0;
  m_currentTransmissionIntensity = 0.0;
  m_txop->SetAifsn (2);
}

bool
ODcf::SendSubsequentPacketInThisBurstIfPossible ()
{
  NS_LOG_FUNCTION (this);

  Ptr<const Packet> packet = m_currentQueue->Peek ();
  if (packet == 0)
    {
      NS_LOG_LOGIC ("No packets in the Media Access Queue");
      FinalizeBurst ();
    }
  else
    {
      // calculated by using the previous MAC header
      uint32_t msduTransmissionDurationInSlot = m_txop->MsduTransmissionDurationInSlotFor (packet);

      NS_LOG_INFO ("holdingDurationInSlot=" << m_currentHoldingDurationInSlot
                                            << " msduTransmissionDurationInSlot="
                                            << msduTransmissionDurationInSlot);

      if (m_currentHoldingDurationInSlot >= msduTransmissionDurationInSlot)
        {
          NS_LOG_LOGIC ("Holding duration is greater than or equal to that of a single packet");

          Mac48Address to;
          Ptr<Packet> packet = m_currentQueue->Dequeue (to);
          if (packet != 0)
            {
              SendSubsequentPacketInThisBurst (packet, to);
              return true;
            }
        }
      else
        {
          NS_LOG_LOGIC ("Holding duration is less than that of a single packet.");
          FinalizeBurst ();
        }
    }

  return false;
}

void
ODcf::SendFirstPacketInThisBurst (Ptr<Packet> packet, const Mac48Address to)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (packet != 0);

  m_isHolding = true;

  uint32_t minCw = GetMinCw ();
  NS_ASSERT (minCw <= m_maxCw);

  // normal access
  m_txop->SetMinCw (minCw);
  m_txop->SetMaxCw (m_maxCw);
  m_txop->SetAifsn (2);
  m_txop->StartBackoff ();

  m_mac->EnqueueToTxop (packet, to);
}

void
ODcf::SendSubsequentPacketInThisBurst (Ptr<Packet> packet, const Mac48Address to)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (packet != 0);

  m_isHolding = true;

  // immediate access
  m_txop->SetMinCw (0);
  m_txop->SetMaxCw (m_maxCw);
  m_txop->SetAifsn (0);
  m_txop->StartBackoff ();

  m_mac->EnqueueToTxop (packet, to);
}

void
ODcf::NotifyMediaAccessQueueHasPacket (Ptr<ODcfQueue> queue)
{
  NS_LOG_FUNCTION (this);

  if (!m_isHolding)
    {
      NS_LOG_LOGIC ("Channel is not holded");

      Mac48Address to;
      Ptr<Packet> packet = queue->Dequeue (to);
      NS_ASSERT (packet != 0);
      m_currentQueue = queue;

      SendFirstPacketInThisBurst (packet, to);
    }
}

void
ODcf::NotifyTransmissionSuccess (uint32_t succCw)
{
  NS_LOG_FUNCTION (this);

  NS_ASSERT (m_currentQueue != 0 && m_isHolding == true);

  NS_LOG_LOGIC ("odcf transmission success");

  UpdateHoldingDuration (succCw);
  if (!SendSubsequentPacketInThisBurstIfPossible ())
    {
      NS_LOG_LOGIC ("Subsequent back-to-back transmission is not possible");
      NS_ASSERT (m_currentQueue == 0);

      m_currentQueue = LinkSchedule ();

      if (m_currentQueue != 0)
        {
          NS_LOG_LOGIC ("Restart channel access after link schedule success");

          Mac48Address to;
          Ptr<Packet> packet = m_currentQueue->Dequeue (to);
          NS_ASSERT (packet != 0);

          SendFirstPacketInThisBurst (packet, to);
        }
    }
}

void
ODcf::NotifyTransmissionFailure ()
{
  NS_LOG_FUNCTION (this);

  NS_ASSERT (m_isHolding == true);

  NS_LOG_LOGIC ("odcf transmission failure");

  FinalizeBurst ();

  NS_ASSERT (m_currentQueue == 0);

  m_currentQueue = LinkSchedule ();

  if (m_currentQueue != 0)
    {
      NS_LOG_LOGIC ("Restart channel access after link schedule failure");

      Mac48Address to;
      Ptr<Packet> packet = m_currentQueue->Dequeue (to);
      NS_ASSERT (packet != 0);

      SendFirstPacketInThisBurst (packet, to);
    }
}

void
ODcf::NotifyCurrentMsduTransmissionDurationInSlot (uint32_t nSlots)
{
  NS_LOG_FUNCTION (this << nSlots);

  m_formerMsduTransmissionDurationInSlot = nSlots;
}

Ptr<const Packet>
ODcf::PeekNextPacket ()
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (m_currentQueue != 0);

  Ptr<const Packet> packet = m_currentQueue->Peek ();
  if (packet != 0)
    {
      if (WillBeFirstImmediateAccess ())
        {
          NS_ASSERT (m_currentHoldingDurationInSlot == 0);
          NS_LOG_LOGIC ("Next transmission is the first first immediate access in this burst");

          return packet;
        }

      uint32_t nextMsduTransmissionDurationInSlot =
          m_txop->MsduTransmissionDurationInSlotFor (packet);

      NS_LOG_INFO ("holdingDurationInSlot="
                   << m_currentHoldingDurationInSlot << " m_formerMsduTransmissionDurationInSlot="
                   << m_formerMsduTransmissionDurationInSlot
                   << " nextMsduTransmissionDurationInSlot=" << nextMsduTransmissionDurationInSlot);

      if (m_currentHoldingDurationInSlot >=
          m_formerMsduTransmissionDurationInSlot + nextMsduTransmissionDurationInSlot)
        {
          NS_LOG_LOGIC ("Holding duration is greater than or equal to that for transmitting "
                        "current and next packets");

          return packet;
        }
    }
  else
    {
      NS_LOG_LOGIC ("No packets in the Media Access Queue");
    }

  return 0;
}

Ptr<ODcfQueue>
ODcf::LinkSchedule ()
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (m_currentQueue == 0);

  Ptr<ODcfQueue> tmpQueue = 0, selectedQueue = 0;

  uint32_t maxMediaAccessQueueSize = 0, tmpMediaAccessQueueSize = 0;
  for (Iterator iterator = m_queues.begin (); iterator != m_queues.end (); iterator++)
    {
      tmpQueue = *iterator;

      tmpMediaAccessQueueSize = tmpQueue->GetMediaAccessQueueSize ();
      if (maxMediaAccessQueueSize < tmpMediaAccessQueueSize)
        {
          maxMediaAccessQueueSize = tmpMediaAccessQueueSize;
          selectedQueue = tmpQueue;
        }
    }

  if (selectedQueue != 0)
    {
      return selectedQueue;
    }
  return 0;
}

double
ODcf::GetTransmissionAggressiveness (uint32_t mediaAccessQueueSize) const
{
  NS_ASSERT (mediaAccessQueueSize <= m_maxQ);
  NS_ASSERT (m_b > 0);

  uint32_t Q = std::max (m_minQ, mediaAccessQueueSize);

  return m_b * Q;
}

Time
ODcf::GetSourceInterval (uint32_t mediaAccessQueueSize) const
{
  NS_ASSERT (m_V > 0);

  return Seconds (GetTransmissionAggressiveness (mediaAccessQueueSize) / m_V);
}

ODcf::Mode
ODcf::GetMode () const
{
  return m_mode;
}

Mac48Address
ODcf::GetAddress () const
{
  if (m_mac != 0)
    {
      return m_mac->GetAddress ();
    }
  return Mac48Address ("00:00:00:00:00:00");
}

} // namespace ns3