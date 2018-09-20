#!/bin/bash
signal=KILL

sleep_a_while () {
    let "mins = ($RANDOM % 50) + 2"
    echo "Sleeping " $mins " minutes"
    sleep ${mins}m 
}

while true; do
    # Note: command launched in background:
    nohup $1 & 

    # Save PID of command just launched:
    last_pid=$!
    echo "Last PID: " $last_pid " !!!!!!!!!!!!!!!!!!!!!!!!!!!"
    # Sleep for a while:
    sleep_a_while

    # See if the command is still running, and kill it and sleep more if it is:
    if ps -p $last_pid -o comm= | grep -qs '^neuroglancer$'; then
        kill -$signal $last_pid 2> /dev/null
        sleep_a_while
    fi

    # Go back to the beginning and launch the command again
done
