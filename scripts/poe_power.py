import subprocess

def get_nodename(node_id):
    switcher = {
        11: "chasseral-1",
        21: "vully-1",
        38: "chaumont-8"
    }
    return switcher.get(node_id, "Invalid node id!")

if __name__ == '__main__':
    p = subprocess.Popen(['./poe', '-j'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    print(out)
