#include "random-app-helper.h"
#include "ns3/random-generator.h"

namespace ns3 {

RandomAppHelper::RandomAppHelper (std::string protocol, Address remote)
{
  m_factory.SetTypeId (RandomGenerator::GetTypeId ());
  m_protocol = protocol;
  m_remote = remote;
}

void
RandomAppHelper::SetAttribute (std::string name, const AttributeValue &value)
{
  m_factory.Set (name, value);
}

ApplicationContainer
RandomAppHelper::Install (NodeContainer nodes)
{
  ApplicationContainer applications;
  for (NodeContainer::Iterator i = nodes.Begin (); i != nodes.End (); ++i)
    {
      Ptr<RandomGenerator> app = m_factory.Create<RandomGenerator> ();
      app->SetRemote (m_protocol, m_remote);
      (*i)->AddApplication (app);
      applications.Add (app);
    }
  return applications;
}

} // namespace ns3