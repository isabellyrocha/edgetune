import subprocess

def set_cores(cores):
    command = "ps -x | grep hyperband_onefold | awk '{print $1}' | while read line ; do sudo taskset -cp -pa 0-%d $line; done" % (int(cores)-1)
    subprocess.Popen(["ssh", "eiger-1.maas", command], shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Changed number of cores to %d... " % cores)

def set_memory(memory):
    #command = "ulimit -Sv %d000000" % memory
    #subprocess.Popen(["ulimit -Sv %d000000" % memory], shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #resource.setrlimit(resource.RLIMIT_VMEM, 1000000)
    subprocess.Popen('ulimit -Sv %d000000' % memory, shell=True)
    print("Changed memory to %dG... " % memory)
