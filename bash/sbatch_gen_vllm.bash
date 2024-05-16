#SBATCH --job-name=dpo
#SBATCH --account=GOV112004
#SBATCH --partition=gpNCHC_H100
#SBATCH --output=O-%x.log
#SBATCH --nodes=1 # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:8                # number of GPUs per node
#SBATCH --cpus-per-task=32         # number of cores per tasks

######################
### Set enviroment ###
######################
export GPUS_PER_NODE=8
######################
export CMD=bash "bash/gen_vllm.bash"
srun $CMD