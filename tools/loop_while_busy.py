import gpustat
import time
from datetime import datetime
import argparse


POWER_THRES = 25
QUERY_INTERVAL = 10
WAIT_ITER = 30

watched_gpus = None
requied_gpu_cnt = None
num_gpus = len(gpustat.new_query().jsonify()['gpus'])


# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--watched_gpus", type=str, default="", help="comma separated list of gpu ids to watch")
parser.add_argument("--requied_gpu_cnt", type=int, default=0, help="")
args = parser.parse_args()
if args.watched_gpus:
    print(f"watching {args.watched_gpus}")
    watched_gpus = [int(gid) for gid in args.watched_gpus.split(",")]
else:
    watched_gpus = list(range(num_gpus))
if args.requied_gpu_cnt > 0:
    requied_gpu_cnt = args.requied_gpu_cnt
else:
    requied_gpu_cnt = len(watched_gpus)


# loop
curr_iter = 0
last_ready_iter = [-1 for _ in watched_gpus]
ready_gpus = []
while True:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    curr_iter += 1
    # check stats
    stats = gpustat.new_query().jsonify()['gpus']
    for i, gpuid in enumerate(watched_gpus):
        s = stats[gpuid]
        if s['power.draw'] < POWER_THRES:
            last_ready_iter[i] = curr_iter if last_ready_iter[i] == -1 else last_ready_iter[i]
        else:
            last_ready_iter[i] = -1
    # count ready gpus
    ready_gpus = []
    for i, lst in enumerate(last_ready_iter):
        if lst != -1 and curr_iter - lst >= WAIT_ITER:
            ready_gpus.append(watched_gpus[i])
    # check and return
    if len(ready_gpus) >= requied_gpu_cnt:
        break
    # print stats
    print(f"[{timestamp}] watching={watched_gpus}, ready_gpus={ready_gpus}, last_ready_iter={last_ready_iter}, curr_iter={curr_iter}")
    gpustat.print_gpustat()
    time.sleep(QUERY_INTERVAL)

print("gpus are ready now")
print(",".join([str(i) for i in ready_gpus]))
