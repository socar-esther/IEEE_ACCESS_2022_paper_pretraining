export PYTHONPATH=$PYTHONPATH
nohup python3.7 -u -m main_upstream 1>logs/upstream_logs.json 2>&1 &