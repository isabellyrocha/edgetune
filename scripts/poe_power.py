from influxdb import InfluxDBClient
from datetime import datetime
import subprocess
import time
import json
import argparse

def get_nodename(node_id):
    switch = {
        11: "chasseral",
        21: "vully",
        38: "chaumont"
    }
    return switch.get(node_id, None)

def get_json(nodename, timestamp, power):
    json_body = [{
        "measurement": "power/node_utilization",
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
    parser = argparse.ArgumentParser(description='POE power reader.')
    parser.add_argument('-p', '--port', type=int, default=11, help='Port.')
    parser.add_argument('-f', '--frequency', type=int, default=1, help='Port.')
    args = parser.parse_args()

    client = InfluxDBClient('localhost', 8086, 'root', 'root', 'power')
    while True:
        timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        p = subprocess.Popen(['./poe', '-p', str(args.port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        print(out.decode('utf-8'))
        array = out.decode('utf-8').split(",")
        nodename = get_nodename(args.port)
        json_body = get_json(nodename, timestamp, float(array[3])) 
        client.write_points(json_body)
#        time.sleep(1)
