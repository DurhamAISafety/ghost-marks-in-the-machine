# Cluster Deployment Instructions

## 1. Transfer Files
Transfer the following files to your cluster directory (e.g., using `scp` or `rsync`):
- `pipeline.py`
- `requirements.txt`
- `run_cluster.sh`
- `.env` (if you have your HF_TOKEN there, otherwise export it)

## 2. Run on Cluster
SSH into your cluster node (or submit a job).

### Interactive Mode (e.g., `srun --pty ...` or direct SSH)
```bash
# Make script executable
chmod +x run_cluster.sh

# Run
./run_cluster.sh
```

### Slurm Job (Batch)
Create a file named `submit_job.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=synthid-eval
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Load modules if needed (e.g., cuda)
# module load cuda/11.8

./run_cluster.sh
```

Then submit:
```bash
sbatch submit_job.slurm
```

## Troubleshooting
- **Jinja2 Error**: If you see `module 'jinja2' has no attribute 'pass_eval_context'`, it means an old version of Jinja2 is installed. The `run_cluster.sh` script creates a fresh virtual environment `.venv` and installs `jinja2>=3.1.0` to fix this.
