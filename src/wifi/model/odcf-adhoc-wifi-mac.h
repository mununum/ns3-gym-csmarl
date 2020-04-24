#ifndef ODCF_ADHOC_WIFI_MAC_H
#define ODCF_ADHOC_WIFI_MAC_H

#include "adhoc-wifi-mac.h"
#include "odcf.h"

#include "amsdu-subframe-header.h"

namespace ns3 {

class ODcfTxop;

class ODcfAdhocWifiMac : public AdhocWifiMac
{
public:
  static TypeId GetTypeId (void);

  ODcfAdhocWifiMac ();
  virtual ~ODcfAdhocWifiMac ();

  virtual void Enqueue (Ptr<const Packet> packet, Mac48Address to);  // MYTODO virtual?
  virtual void EnqueueToTxop (Ptr<const Packet> packet, Mac48Address to);
  virtual void SetWifiRemoteStationManager (Ptr<WifiRemoteStationManager> stationManager);

  uint32_t GetMAQSize ();

protected:
  // void DoStart ();
  // void DoDispose ();

  Ptr<ODcfTxop> m_odcfTxop;
  Ptr<ODcf> m_odcf;
};

} // namespace ns3

#endif /* ODCF_ADHOC_WIFI_MAC_H */