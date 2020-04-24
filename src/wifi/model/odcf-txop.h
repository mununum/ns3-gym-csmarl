/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005 INRIA
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#ifndef ODCF_TXOP_H
#define ODCF_TXOP_H

#include "txop.h"
#include "mac-low-transmission-parameters.h"
#include "wifi-mac-header.h"

#include "odcf.h"

namespace ns3 {

class Packet;
class ChannelAccessManager;
class MacTxMiddle;
class MacLow;
class WifiMode;
class WifiMacQueue;
class WifiMacQueueItem;
class UniformRandomVariable;
class CtrlBAckResponseHeader;
class WifiRemoteStationManager;

class ODcf;
class ODcfAdhocWifiMac;

/**
 * \brief Handle packet fragmentation and retransmissions
 * for data and management frames.
 * \ingroup wifi
 *
 * This class implements the packet fragmentation and
 * retransmission policy for data and management frames.
 * It uses the ns3::MacLow and ns3::ChannelAccessManager helper
 * classes to respectively send packets and decide when
 * to send them. Packets are stored in a ns3::WifiMacQueue
 * until they can be sent.
 *
 * The policy currently implemented uses a simple fragmentation
 * threshold: any packet bigger than this threshold is fragmented
 * in fragments whose size is smaller than the threshold.
 *
 * The retransmission policy is also very simple: every packet is
 * retransmitted until it is either successfully transmitted or
 * it has been retransmitted up until the ssrc or slrc thresholds.
 *
 * The rts/cts policy is similar to the fragmentation policy: when
 * a packet is bigger than a threshold, the rts/cts protocol is used.
 */

class ODcfTxop : public Txop // MYTODO is this correct?
{
public:
  /// allow DcfListener class access
  friend class DcfListener;
  /// allow MacLowTransmissionListener class access
  friend class MacLowTransmissionListener;

  ODcfTxop ();
  ODcfTxop (Ptr<ODcfAdhocWifiMac> mac);
  virtual ~ODcfTxop ();

  void SetODcf (Ptr<ODcf> odcf);
  uint32_t MsduTransmissionDurationInSlotFor (Ptr<const Packet> packet);

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);


  /* Event handlers */
  /**
   * Event handler when a CTS timeout has occurred.
   */
  virtual void MissedCts (void);
  /**
   * Event handler when an ACK is received.
   */
  virtual void GotAck (void);
  /**
   * Event handler when an ACK is missed.
   */
  virtual void MissedAck (void);

  void StartBackoff (void); // MYTODO do we need this?

protected:
  ///< ChannelAccessManager associated class
  friend class ChannelAccessManager;

  virtual void DoDispose (void);
  virtual void DoInitialize (void);

  /* dcf notifications forwarded here */
  /**
   * Notify the DCF that access has been granted.
   */
  virtual void NotifyAccessGranted (void);

  uint32_t CurrentMsduTransmissionDurationInSlot ();
  Time CalculateNavFor (Ptr<const Packet> packet);


  Ptr<ODcf> m_odcf;
  Ptr<ODcfAdhocWifiMac> m_mac;
};

} //namespace ns3

#endif /* ODCF_TXOP_H */
