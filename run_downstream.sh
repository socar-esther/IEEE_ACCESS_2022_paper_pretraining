export PYTHONPATH=$PYTHONPATH
nohup python3.7 -u -m main_downstream_car_class_classifier 1>logs/car_class_downatream_logs.json 2>&1 &