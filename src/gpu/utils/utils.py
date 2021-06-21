import subprocess
import psutil

def set_training_cores(cores):
    command = "ps -x | grep Train | awk '{print $1}' | while read line ; do sudo taskset -cp -pa 0-%d $line; done" % (int(cores)-1)
    subprocess.Popen(["ssh", "jolly.maas", command], shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Changed number of cores to %d... " % cores)

def set_inference_cores(cores):
    #command = "ps -x | awk '{print $1}' | while read line ; do taskset -cp -pa 0-%d $line; done" % (int(cores)-1)
    #subprocess.Popen(["ssh", "jolly.maas", command], shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for p in psutil.process_iter(attrs=["pid", "name"]):
         command = "taskset -cp -pa 0-%d %s" % (int(cores)-1, str(p.pid))
         process =  subprocess.Popen(command.split(" "), shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
         stdout, stderr = process.communicate()
         print(stdout)
    print("Changed number of cores to %d... " % cores)

def set_memory(memory):
    command = "ulimit -Sv %d000000" % memory
    subprocess.Popen(["ulimit -Sv %d000000" % memory], shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Changed memory to %dG... " % memory)
