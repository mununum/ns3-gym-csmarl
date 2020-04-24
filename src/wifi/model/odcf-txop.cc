#include "ns3/assert.h"
#include "ns3/log.h"
#include "ns3/pointer.h"
#include "ns3/simulator.h"
#include "ns3/random-variable-stream.h"
#include "odcf-txop.h"
#include "channel-access-manager.h"
#include "wifi-mac-queue.h"
#include "wifi-mac-trailer.h"
#include "wifi-phy.h"
#include "mac-tx-middle.h"
#include "mac-low.h"
#include "wifi-remote-station-manager.h"

#undef NS_LOG_APPEND_CONTEXT
#define NS_LOG_APPEND_CONTEXT                               \
  if (m_low != 0)                                           \
    {                                                       \
      std::clog << "[mac=" << m_low->GetAddress () << "] "; \
    }

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ODcfTxop");

NS_OBJECT_ENSURE_REGISTERED (ODcfTxop);

TypeId
ODcfTxop::GetTypeId (void)
{
  static TypeId tid =
      TypeId ("ns3::ODcfTxop")
          .SetParent<Txop> ()
          .SetGroupName ("Wifi")
          .AddConstructor<ODcfTxop> ();
  return tid;
}

ODcfTxop::ODcfTxop () : Txop ()
{
  NS_LOG_FUNCTION (this);
}

ODcfTxop::ODcfTxop (Ptr<ODcfAdhocWifiMac> mac) : Txop (), m_mac (mac)
{
  NS_LOG_FUNCTION (this);
}

ODcfTxop::~ODcfTxop ()
{
  NS_LOG_FUNCTION (this);
}

void
ODcfTxop::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  m_queue = 0;
  m_low = 0;
  m_stationManager = 0;
  m_rng = 0;
  m_txMiddle = 0;
  m_channelAccessManager = 0;
}

void
ODcfTxop::SetODcf (Ptr<ODcf> odcf)
{
  m_odcf = odcf;
}

void
ODcfTxop::StartBackoff (void)
{
  NS_LOG_FUNCTION (this);
  // NS_ASSERT (m_dcf != 0);

  // MYTODO do we need this function?

  uint32_t nSlots = m_rng->GetInteger (0, GetCw ());
  StartBackoffNow (nSlots);
}

void
ODcfTxop::DoInitialize ()
{
  NS_LOG_FUNCTION (this);
  ResetCw ();
  StartBackoffNow (m_rng->GetInteger (0, GetCw ()));
}

uint32_t
ODcfTxop::CurrentMsduTransmissionDurationInSlot ()
{
  return MsduTransmissionDurationInSlotFor (m_currentPacket);
}

uint32_t
ODcfTxop::MsduTransmissionDurationInSlotFor (Ptr<const Packet> packet)
{
  NS_LOG_FUNCTION (this);

  uint32_t msduSize = packet->GetSize (); // MSDU size

  Mac48Address to = m_currentHdr.GetAddr1 ();

  WifiTxVector dataTxVector = m_stationManager->GetDataTxVector (to, &m_currentHdr, packet);

  Ptr<WifiPhy> wifiPhy = GetLow ()->GetPhy ();
  uint16_t frequency = wifiPhy->GetFrequency ();
  uint32_t durationInMicroSeconds =
      wifiPhy->GetPayloadDuration (msduSize, dataTxVector, frequency).GetMicroSeconds ();

  uint32_t slotTimeInMicroSeconds = GetLow ()->GetSlotTime ().GetMicroSeconds ();
  uint32_t durationInSlots = 0;
  if (durationInMicroSeconds % slotTimeInMicroSeconds != 0) // ceiling
    {
      durationInSlots = 1;
    }
  durationInSlots += durationInMicroSeconds / slotTimeInMicroSeconds;

  return durationInSlots;
}

Time
ODcfTxop::CalculateNavFor (Ptr<const Packet> packet)
{
  NS_LOG_FUNCTION (this);

  uint32_t msduSize = packet->GetSize (); // MSDU size
  WifiMacTrailer fcs;
  WifiMacHeader ack;
  ack.SetType (WIFI_MAC_CTL_ACK);

  Mac48Address to = m_currentHdr.GetAddr1 (); // use current header

  uint32_t mpduSize = m_currentHdr.GetSize () + msduSize + fcs.GetSerializedSize ();
  WifiTxVector dataTxVector = m_stationManager->GetDataTxVector (to, &m_currentHdr, packet);
  Ptr<WifiPhy> wifiPhy = GetLow ()->GetPhy ();
  uint16_t frequency = wifiPhy->GetFrequency ();
  Time dataDuration = wifiPhy->CalculateTxDuration (mpduSize, dataTxVector, frequency);

  uint32_t ackSize = ack.GetSize () + fcs.GetSerializedSize ();
  WifiTxVector ackTxVector = m_stationManager->GetAckTxVector (to, dataTxVector.GetMode ());
  Time ackDuration = wifiPhy->CalculateTxDuration (ackSize, ackTxVector, frequency);

  Time sifs = GetLow ()->GetSifs ();

  Time nav = sifs + ackDuration + sifs + dataDuration + sifs + ackDuration;

  return nav;
}

void
ODcfTxop::NotifyAccessGranted (void)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (m_accessRequested);
  m_accessRequested = false;
  if (m_currentPacket == 0)
    {
      if (m_queue->IsEmpty ())
        {
          NS_LOG_DEBUG ("queue empty");
          return;
        }
      Ptr<WifiMacQueueItem> item = m_queue->Dequeue ();
      NS_ASSERT (item != 0);
      m_currentPacket = item->GetPacket ();
      m_currentHdr = item->GetHeader ();
      m_currentHdr.m_packet = (void *) PeekPointer (m_currentPacket);
      NS_ASSERT (m_currentPacket != 0);
      uint16_t sequence = m_txMiddle->GetNextSequenceNumberFor (&m_currentHdr);
      m_currentHdr.SetSequenceNumber (sequence);
      m_stationManager->UpdateFragmentationThreshold ();
      m_currentHdr.SetFragmentNumber (0);
      m_currentHdr.SetNoMoreFragments ();
      m_currentHdr.SetNoRetry ();
      m_fragmentNumber = 0;
      NS_LOG_DEBUG ("dequeued size=" << m_currentPacket->GetSize ()
                                     << ", to=" << m_currentHdr.GetAddr1 ()
                                     << ", seq=" << m_currentHdr.GetSequenceControl ());
    }

  m_odcf->NotifyCurrentMsduTransmissionDurationInSlot (CurrentMsduTransmissionDurationInSlot ());

  if (m_currentHdr.GetAddr1 ().IsGroup ())
    {
      m_currentParams.DisableRts ();
      m_currentParams.DisableAck ();
      m_currentParams.DisableNextData ();
      NS_LOG_DEBUG ("tx broadcast");
      GetLow ()->StartTransmission (m_currentPacket, &m_currentHdr, m_currentParams, this);

      m_odcf->NotifyTransmissionSuccess (GetCw ());
    }
  else
    {
      m_currentParams.EnableAck ();
      if (NeedFragmentation ())
        {
          WifiMacHeader hdr;
          Ptr<Packet> fragment = GetFragmentPacket (&hdr);
          if (IsLastFragment ())
            {
              NS_LOG_DEBUG ("fragmenting last fragment size=" << fragment->GetSize ());
              m_currentParams.DisableNextData ();
            }
          else
            {
              NS_LOG_DEBUG ("fragmenting size=" << fragment->GetSize ());
              m_currentParams.EnableNextData (GetNextFragmentSize ());
            }
          GetLow ()->StartTransmission (fragment, &hdr, m_currentParams, this);
        }
      else
        {

          // NAV setting only for unicast
          Ptr<const Packet> packet = m_odcf->PeekNextPacket ();
          if (packet != 0)
            {
              // MYTODO this function does not exist in this version!
              // this function is called for back-to-back transmission (Sec V-D-3 in O-DCF paper)
              // m_currentParams.EnableOverrideDurationId (CalculateNavFor (packet));
            }

          m_currentParams.DisableNextData ();
          GetLow ()->StartTransmission (m_currentPacket, &m_currentHdr, m_currentParams, this);
        }
    }
}

void
ODcfTxop::MissedCts (void)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_DEBUG ("missed cts");
  if (!NeedRtsRetransmission (m_currentPacket, m_currentHdr))
    {
      NS_LOG_DEBUG ("Cts Fail");
      m_stationManager->ReportFinalRtsFailed (m_currentHdr.GetAddr1 (), &m_currentHdr);
      if (!m_txFailedCallback.IsNull ())
        {
          m_txFailedCallback (m_currentHdr);
        }

      m_odcf->NotifyTransmissionFailure ();

      //to reset the dcf.
      m_currentPacket = 0;
      ResetCw ();
    }
  else
    {
      UpdateFailedCw ();
      StartBackoffNow (m_rng->GetInteger (0, GetCw ())); // MD
    }
  RestartAccessIfNeeded ();
}

void
ODcfTxop::GotAck (void)
{
  NS_LOG_FUNCTION (this);
  if (!NeedFragmentation () || IsLastFragment ())
    {
      NS_LOG_DEBUG ("got ack. tx done.");
      if (!m_txOkCallback.IsNull ())
        {
          m_txOkCallback (m_currentHdr);
        }

      m_odcf->NotifyTransmissionSuccess (GetCw ());

      /* we are not fragmenting or we are done fragmenting
       * so we can get rid of that packet now.
       */
      m_currentPacket = 0;
      ResetCw ();
      StartBackoffNow (m_rng->GetInteger (0, GetCw ()));
      RestartAccessIfNeeded ();
    }
  else
    {
      NS_LOG_DEBUG ("got ack. tx not done, size=" << m_currentPacket->GetSize ());
    }
}

void
ODcfTxop::MissedAck (void)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_DEBUG ("missed ack");
  if (!NeedDataRetransmission (m_currentPacket, m_currentHdr))
    {
      NS_LOG_DEBUG ("Ack Fail");
      m_stationManager->ReportFinalDataFailed (m_currentHdr.GetAddr1 (), &m_currentHdr,
                                               m_currentPacket->GetSize ());
      if (!m_txFailedCallback.IsNull ())
        {
          m_txFailedCallback (m_currentHdr);
        }

      m_odcf->NotifyTransmissionFailure ();

      //to reset the dcf.
      m_currentPacket = 0;
      ResetCw ();
    }
  else
    {
      NS_LOG_DEBUG ("Retransmit");
      m_currentHdr.SetRetry ();
      UpdateFailedCw ();
      StartBackoffNow (m_rng->GetInteger (0, GetCw ()));
    }
  RestartAccessIfNeeded ();
}

} // namespace ns3