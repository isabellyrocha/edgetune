from influxdb import InfluxDBClient
from pathlib import Path
from pathlib import Path
from datetime import datetime
import time
import argparse
import numpy

def str_to_tstp(time_str:str):
    return int(time.mktime(datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S').timetuple()))

def query_power_data(node_name, start, end):
    client = InfluxDBClient('localhost', 8086, 'root', 'root', 'power')
    result = client.query('SELECT max(value) '
                'FROM "power/node_utilization" '
                'WHERE nodename =~ /%s/ AND '
                '%d000000000 <= time AND '
                'time <= %d000000000 '
                'group by time(1s) fill(previous)' % (node_name, start, end))
    return list(result.get_points(measurement='power/node_utilization'))


def energy(nodes, start, end):
    energy = 0
    #print(start)
    #print(end)
    for node_name in nodes: #['eiger-1', 'eiger-2', 'eiger-4', 'eiger-5']:
        points = query_power_data(node_name, start, end)
        values = []

        last = 0
        for i in range(len(points)):
            value = points[i]['max']
            if not value == None:
                for j in range(last,i):
                    values.append(value)
                last = i
        for i in range(last, len(points)):
            values.append(value)
        energy += (numpy.trapz(values))
    return energy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--node", type=str, help="Node name.", default="scopi")
    args, _ = parser.parse_known_args()

    with open("%s/edgetune/exps/results/%s.out" % (Path.home(), args.node)) as f:
        line = f.readline()
        while line:
            network,dataset,start,end = line.rstrip().split(",")
            start_s = str_to_tstp(start.replace("Z", "").replace("T", " "))
            end_s = str_to_tstp(end.replace("Z", "").replace("T", " "))
            node_duration = (end_s - start_s)
            node_energy = energy(args.node,start_s, end_s)
            print("%s,%s,%s,%d,%f" % (args.node, network, dataset, node_duration, node_energy))
            line = f.readline()
