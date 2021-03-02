import sys
import docker
client = docker.from_env()

if len(sys.argv) <= 1:
    print("usage: {} [container_name]".format(sys.argv[0]))
    exit(1)
    
name = sys.argv[1]
running = client.containers.get(name)

# get model
with open("{}.tar".format(name), "wb") as f:
    bits, stat = running.get_archive("/root/ray_results/")
    print(stat)
    for chunk in bits:
        f.write(chunk)
