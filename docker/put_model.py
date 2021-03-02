import sys
import docker
client = docker.from_env()

if len(sys.argv) <= 1:
    print("usage: {} [container_name]".format(sys.argv[0]))
    exit(1)

name = sys.argv[1]

# test container
testing = client.containers.run("ns3-gym-csmarl:src", "tail -f /dev/null", name="tester",
                                auto_remove=True, detach=True, runtime="nvidia", shm_size="80G")
# put trained model
with open("{}.tar".format(name), "rb") as f:
    data = f.read()
    testing.put_archive("/root/", data)
