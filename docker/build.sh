docker build -t ns3-gym-csmarl:base -f Dockerfile_base .
docker build -t ns3-gym-csmarl:git -f Dockerfile_from_github .
docker build -t ns3-gym-csmarl:src -f Dockerfile_from_source ..