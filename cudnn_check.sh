#!/bin/bash -l
#SBATCH --job-name=cudnn_check
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition=csgpu
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:10:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mercy.edoho@ucdconnect.ie

module purge
module load anaconda3
conda activate torch_v100_py310

cd ~/TCN_SSL_GA

python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.version.cuda)
print('cuDNN:', torch.backends.cudnn.version())
print('GPU:', torch.cuda.get_device_name(0))

# Test with deterministic mode (as set_seed does)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from tcn_utils import TCN, set_seed
set_seed(42)
model = TCN(6, 32, 7, 0.35).cuda()
x = torch.randn(64, 1, 2500).cuda()
try:
    out = model(x)
    print('Forward pass (deterministic): OK, output shape:', out.shape)
except RuntimeError as e:
    print('Forward pass (deterministic) FAILED:', e)

# Test without deterministic mode
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
try:
    out = model(x)
    print('Forward pass (non-deterministic): OK, output shape:', out.shape)
except RuntimeError as e:
    print('Forward pass (non-deterministic) FAILED:', e)
"
