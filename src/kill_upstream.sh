PID=`ps -ef | grep “main_upstream” | grep -v grep | awk ‘{print $2}’`
kill -9 $PID