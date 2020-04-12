#ifndef RANDOM_GENERATOR_H
#define RANDOM_GENERATOR_H

#include "ns3/application.h"
#include "ns3/event-id.h"
#include "ns3/random-variable-stream.h"
#include "ns3/ptr.h"
#include "ns3/ipv4-address.h"

namespace ns3 {

class Socket;

class RandomGenerator : public Application
{
public:
  static TypeId GetTypeId (void);
  RandomGenerator ();
  void SetDelay (Ptr<RandomVariableStream> delay);
  void SetSize (Ptr<RandomVariableStream> size);
  void SetRemote (std::string socketType, Address remote);

private:
  virtual void StartApplication (void);
  virtual void StopApplication (void);
  void DoGenerate (void);

  TypeId m_socketType;
  Address m_peerAddress;

  Time m_lastMod;
  Time m_sampledModDelay;

  uint8_t m_mode;
  Ptr<RandomVariableStream> m_delay1;
  Ptr<RandomVariableStream> m_delay2;
  Ptr<RandomVariableStream> m_modDelay;

  Ptr<RandomVariableStream> m_size;
  Ptr<Socket> m_socket;
  EventId m_next;
};

} // namespace ns3

#endif /* RANDOM_GENERATOR_H */