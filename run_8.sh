nohup taskset -c 0 python inference/worker.py --queue_name deepalign0 --lease_seconds 200 & 
nohup taskset -c 1 python inference/worker.py --queue_name deepalign0 --lease_seconds 200 &
nohup taskset -c 2 python inference/worker.py --queue_name deepalign0 --lease_seconds 200 &
nohup taskset -c 3 python inference/worker.py --queue_name deepalign0 --lease_seconds 200 &
nohup taskset -c 4 python inference/worker.py --queue_name deepalign0 --lease_seconds 200 &
nohup taskset -c 5 python inference/worker.py --queue_name deepalign0 --lease_seconds 200 &
nohup taskset -c 6 python inference/worker.py --queue_name deepalign0 --lease_seconds 200 &
nohup taskset -c 7 python inference/worker.py --queue_name deepalign0 --lease_seconds 200 &
