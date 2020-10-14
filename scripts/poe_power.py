from influxdb import InfluxDBClient
from datetime import datetime
import subprocess
import time
import json

def get_nodename(node_id):
    switch = {
        11: "chasseral-1",
        21: "vully-1",
        38: "chaumont-8"
    }
    return switch.get(node_id, None)

def get_json(nodename, timestamp, power):
    json_body = [{
        "measurement": "poe_power/node_utilization",
        "tags": {
            "nodename": nodename,
        },
        "time": timestamp,
        "fields": {
            "value": power
        }
    }]
    return json_body

if __name__ == '__main__':
    client = InfluxDBClient('localhost', 8086, 'root', 'root', 'power')
    while True:
        timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        p = subprocess.Popen(['./poe', '-j'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        array = out.decode('utf-8').splitlines()
        for line in array:
           try:
                info = json.loads(line.lstrip("b\'").rstrip("\'").lstrip())
                nodename = get_nodename(info['interface'])
                if nodename:
                    json_body = get_json(nodename, timestamp, info['consumed']) 
                    client.write_points(json_body)
           except Exception as err:
             print("Error! ", err)
#        time.sleep(1)
