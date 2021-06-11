import subprocess

def set_inference_cores(cores):
    command = "ps -aux | awk '{print $2}' | while read line ; do sudo taskset -cp -pa 0-%d $line; done" % (int(cores)-1)
    subprocess.Popen(["ssh", "nuc-1.maas", command], shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Changed number of cores to %d... " % cores)

def set_memory(memory):
    command = "ulimit -Sv %d000000" % memory
    subprocess.Popen(["ulimit -Sv %d000000" % memory], shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Changed memory to %dG... " % memory)
