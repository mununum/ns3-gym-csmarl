#include "ns3/pointer.h"
#include "ns3/log.h"
#include "ns3/string.h"
#include "ns3/boolean.h"
#include "ns3/trace-source-accessor.h"

// #include "qos-tag.h"
#include "mac-low.h"
#include "channel-access-manager.h"
#include "mac-rx-middle.h"
#include "mac-tx-middle.h"
#include "msdu-aggregator.h"
#include "amsdu-subframe-header.h"
#include "mgt-headers.h"

#include "odcf-adhoc-wifi-mac.h"

#undef NS_LOG_APPEND_CONTEXT
#define NS_LOG_APPEND_CONTEXT                              \
  if (Mac48Address ("00:00:00:00:00:00") != GetAddress ()) \
    {                                                      \
      std::clog << "[mac=" << GetAddress () << "] ";       \
    }

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ODcfAdhocWifiMac");

NS_OBJECT_ENSURE_REGISTERED (ODcfAdhocWifiMac);

TypeId
ODcfAdhocWifiMac::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ODcfAdhocWifiMac")
                          .SetParent<AdhocWifiMac> ()
                          .SetGroupName ("Wifi")
                          .AddConstructor<ODcfAdhocWifiMac> ();
  return tid;
}

ODcfAdhocWifiMac::ODcfAdhocWifiMac ()
{
  NS_LOG_FUNCTION (this);

  m_odcfTxop = CreateObject<ODcfTxop> (this);
  m_odcfTxop->SetMacLow (m_low);
  m_odcfTxop->SetChannelAccessManager (m_channelAccessManager);
  m_odcfTxop->SetAifsn (m_txop->GetAifsn ());
  m_odcfTxop->SetMinCw (m_txop->GetMinCw ());
  m_odcfTxop->SetMaxCw (m_txop->GetMaxCw ());
  m_odcfTxop->SetTxMiddle (m_txMiddle);

  m_odcfTxop->SetTxOkCallback (MakeCallback (&ODcfAdhocWifiMac::TxOk, this));
  m_odcfTxop->SetTxFailedCallback (MakeCallback (&ODcfAdhocWifiMac::TxFailed, this));
  m_odcfTxop->SetTxDroppedCallback (MakeCallback (&ODcfAdhocWifiMac::NotifyTxDrop, this));

  m_odcf = CreateObject<ODcf> (this, m_odcfTxop, m_txop->GetMaxCw ());

  // Let the lower layers know that we are acting in an IBSS
  SetTypeOfStation (ADHOC_STA);
}

ODcfAdhocWifiMac::~ODcfAdhocWifiMac ()
{
  NS_LOG_FUNCTION (this);
}

// void
// ODcfAdhocWifiMac::DoStart ()
// {
//   NS_LOG_FUNCTION (this);

//   m_odcfTxop->Start ();
//   AdhocWifiMac::DoStart ();
// }

// void
// ODcfAdhocWifiMac::DoDispose ()
// {
//   NS_LOG_FUNCTION (this);

//   m_odcfTxop = 0;
//   AdhocWifiMac::DoDispose ();
// }

// return the first queue for debugging purpose
uint32_t
ODcfAdhocWifiMac::GetMAQSize ()
{
  if (m_odcf->m_queues.empty())
    return 0;
  return (*m_odcf->m_queues.begin())->GetMediaAccessQueueSize ();
}

void
ODcfAdhocWifiMac::SetWifiRemoteStationManager (Ptr<WifiRemoteStationManager> stationManager)
{
  NS_LOG_FUNCTION (this << stationManager);

  m_odcfTxop->SetWifiRemoteStationManager (stationManager);
  AdhocWifiMac::SetWifiRemoteStationManager (stationManager);
}

void
ODcfAdhocWifiMac::Enqueue (Ptr<const Packet> packet, Mac48Address to)
{
  NS_LOG_FUNCTION (this << packet << to);

  m_odcf->Enqueue (packet->Copy (), to);
}

void
ODcfAdhocWifiMac::EnqueueToTxop (Ptr<const Packet> packet, Mac48Address to)
{
  NS_LOG_FUNCTION (this << packet << to);

  if (m_stationManager->IsBrandNew (to))
    {
      // In ad hoc mode, we assume that every destination supports al
      // the rates we support.
      if (GetHtSupported () || GetVhtSupported () || GetHeSupported ())
        {
          m_stationManager->AddAllSupportedMcs (to);
        }
      if (GetHtSupported ())
        {
          m_stationManager->AddStationHtCapabilities (to, GetHtCapabilities ());
        }
      if (GetVhtSupported ())
        {
          m_stationManager->AddStationVhtCapabilities (to, GetVhtCapabilities ());
        }
      if (GetHeSupported ())
        {
          m_stationManager->AddStationHeCapabilities (to, GetHeCapabilities ());
        }
      m_stationManager->AddAllSupportedModes (to);
      m_stationManager->RecordDisassociated (to);
    }

  WifiMacHeader hdr;

  //If we are not a QoS STA then we definitely want to use AC_BE to
  //transmit the packet. A TID of zero will map to AC_BE (through \c
  //QosUtilsMapTidToAc()), so we use that as our default here.
  uint8_t tid = 0;

  //For now, a STA that supports QoS does not support non-QoS
  //associations, and vice versa. In future the STA model should fall
  //back to non-QoS if talking to a peer that is also non-QoS. At
  //that point there will need to be per-station QoS state maintained
  //by the association state machine, and consulted here.
  if (GetQosSupported ())
    {
      hdr.SetType (WIFI_MAC_QOSDATA);
      hdr.SetQosAckPolicy (WifiMacHeader::NORMAL_ACK);
      hdr.SetQosNoEosp ();
      hdr.SetQosNoAmsdu ();
      //Transmission of multiple frames in the same TXOP is not
      //supported for now
      hdr.SetQosTxopLimit (0);

      //Fill in the QoS control field in the MAC header
      tid = QosUtilsGetTidForPacket (packet);
      //Any value greater than 7 is invalid and likely indicates that
      //the packet had no QoS tag, so we revert to zero, which will
      //mean that AC_BE is used.
      if (tid > 7)
        {
          tid = 0;
        }
      hdr.SetQosTid (tid);
    }
  else
    {
      hdr.SetType (WIFI_MAC_DATA);
    }

  if (GetHtSupported () || GetVhtSupported () || GetHeSupported ())
    {
      hdr.SetNoOrder (); // explicitly set to 0 for the time being since HT/VHT/HE control field is not yet implemented (set it to 1 when implemented)
    }
  hdr.SetAddr1 (to);
  hdr.SetAddr2 (m_low->GetAddress ());
  hdr.SetAddr3 (GetBssid ());
  hdr.SetDsNotFrom ();
  hdr.SetDsNotTo ();

  if (GetQosSupported ())
    {
      NS_ASSERT_MSG (false, "qos not supported yet");

      //Sanity check that the TID is valid
      NS_ASSERT (tid < 8);
      m_edca[QosUtilsMapTidToAc (tid)]->Queue (packet, hdr);
    }
  else
    {
      NS_LOG_DEBUG ("Sending packet to IFQ");
      m_odcfTxop->Queue (packet, hdr);
    }
}

} // namespace ns3