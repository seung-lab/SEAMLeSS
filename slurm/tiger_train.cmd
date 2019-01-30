TCH -N 1
#SBATCH --job-name=align
#SBATCH --ntasks-per-node=1
#SBATCH --output=bufferlog.txt
#SBATCH --cpus-per-task=11
#SBATCH --gres=gpu:10
#SBATCH -t 72:00:00
#SBATCH --mem 128G
# sends mail when process begins, and
# when it ends. Make sure you difine your email
# address
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=runzhey@cs.princeton.edu

export PATH="/home/runzhey/miniconda3/bin:$PATH"
export LANGUAGE=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
source activate rllab3

cd /home/runzhey/SEAMLeSS/training
./train.py --num_workers 10 --gpu_ids 0,1,2,3,4,5,6,7,8,9 resume test_encodings_all_samples_mGPU_c2e
