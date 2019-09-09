kubectl get pods | grep Running | awk '{print $1}' > name.txt
file="name.txt"
#echo $file

while read line
do
#kubectl logs ${line} | awk 'END {print}'
kubectl logs ${line} | grep 'get_image: '
#kubectl logs ${line} | grep 'Random_write_time:'

done <$file

