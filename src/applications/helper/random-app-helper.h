#ifndef RANDOM_APP_HELPER_H
#define RANDOM_APP_HELPER_H

#include "ns3/random-variable-stream.h"
#include "ns3/ipv4-address.h"
#include "ns3/application-container.h"
#include "ns3/node-container.h"
#include "ns3/object-factory.h"

namespace ns3 {

class RandomAppHelper
{
public:
  RandomAppHelper (std::string protocol, Address remote);
  void SetAttribute (std::string name, const AttributeValue &value);
  ApplicationContainer Install (NodeContainer nodes);
private:
  std::string m_protocol;
  Address m_remote;
  ObjectFactory m_factory;
};

} // namespace ns3

#endif /* RANDOM_APP_HELPER_H */