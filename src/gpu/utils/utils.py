import subprocess

def set_training_cores(cores):
    command = "ps -x | grep Train | awk '{print $1}' | while read line ; do sudo taskset -cp -pa 0-%d $line; done" % (int(cores)-1)
    subprocess.Popen(["ssh", "eiger-1.maas", command], shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Changed number of cores to %d... " % cores)

def set_inference_cores(cores):
    command = "ps -x | grep Inference | awk '{print $1}' | while read line ; do sudo taskset -cp -pa 4-%d $line; done" % (3+int(cores))
    subprocess.Popen(["ssh", "eiger-1.maas", command], shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Changed number of cores to %d... " % cores)

def set_memory(memory):
    #command = "ulimit -Sv %d000000" % memory
    #subprocess.Popen(["ulimit -Sv %d000000" % memory], shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #resource.setrlimit(resource.RLIMIT_VMEM, 1000000)
    subprocess.Popen('ulimit -Sv %d000000' % memory, shell=True)
    print("Changed memory to %dG... " % memory)
