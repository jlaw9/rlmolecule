""" Main starting point
"""

import argparse
import pathlib
import os
import sys
import subprocess

from rlmolecule.config import Config


class Submission_Script:
    def __init__(
            self, hpc_env="eagle", job_name="example", out_file="submit.sh",
            working_dir="", time="4:00:00", account="rlmolecule",
            start_policy_script="", start_rollout_script="",
            policy_env=None, rollout_env=None,
            rollout_nodes=2, rollout_ntasks_per_node=6,
            policy_nodes=1, policy_gpus=2,
            **kwargs):
        """
        *hpc_env*: Computing environment for which a submission script will be 
            generated and submitted. Allowed values: 'eagle'
        *working_dir*: Directory where submission scripts and log files 
            will be stored
        *policy_env*: string to load the environment used for the GPU nodes
        *rollout_env*: string to load the environment used for the CPU nodes
        """
        self.hpc_env = hpc_env
        self.job_name = job_name
        self.out_file = out_file
        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)
        self.time = time
        self.account = account
        self.start_policy_script = start_policy_script
        self.start_rollout_script = start_rollout_script
        self.policy_env = policy_env
        self.rollout_env = rollout_env
        self.rollout_nodes = rollout_nodes
        self.rollout_ntasks_per_node = rollout_ntasks_per_node
        self.policy_nodes = policy_nodes
        self.policy_gpus = policy_gpus

    def setup_hpc_submission_scripts(self, submit=False):
        if self.hpc_env == "eagle":
            submission_str = self.generate_eagle_submission_string()
            command = f"sbatch {self.out_file}"
        else:
            print(f"ERROR: not yet setup to run on '{self.hpc_env}'")
            print("Quitting")
            sys.exit()

        print(f"Writing submissing script to {self.out_file}")
        with open(self.out_file, 'w') as out:
            out.write(submission_str)

        if submit:
            print(f"Running {command}")
            subprocess.check_call(command, shell=True)

    def generate_eagle_submission_string(self):
        submission_str = f"""#!/bin/bash
#SBATCH --account={self.account}
#SBATCH --time={self.time}
#SBATCH --job-name={self.job_name}
# --- Policy Trainer ---
#SBATCH --nodes={self.policy_nodes}
#SBATCH --gres=gpu:{self.policy_gpus}
# --- MCTS Rollouts ---
#SBATCH hetjob
#SBATCH -N {self.rollout_nodes}

export WORKING_DIR={self.working_dir}
export START_POLICY_SCRIPT="$SLURM_SUBMIT_DIR/$JOB/.policy.sh"
export START_ROLLOUT_SCRIPT="$SLURM_SUBMIT_DIR/$JOB/.rollout.sh"

cat << "EOF" > "$START_POLICY_SCRIPT"
#!/bin/bash
{self.policy_env}
python -u {self.start_policy_script}
EOF

cat << "EOF" > "$START_ROLLOUT_SCRIPT"
#!/bin/bash
{self.rollout_env}
python -u {self.start_rollout_script}
EOF

chmod +x "$START_POLICY_SCRIPT" "$START_ROLLOUT_SCRIPT"

srun --pack-group=0 \
     --job-name="az-policy" \
     --output=$WORKING_DIR/gpu.%j.out \
     "$START_POLICY_SCRIPT" &

srun --pack-group=1 \
     --ntasks-per-node=6 \
     --job-name="az-rollout" \
     --output=$WORKING_DIR/mcts.%j.out \
     "$START_ROLLOUT_SCRIPT"
"""
        return submission_str


def run(config_file, submit=False, **kwargs):

    config = Config(config_file, **kwargs)

    if config.hpc_config is not None:
        ss = Submission_Script(**config.hpc_config)
        ss.setup_hpc_submission_scripts(submit=submit)
    else:
        command = f"python -u {config.run_config['start_script']}"
        print(f"Running {command}")
        subprocess.check_call(command, shell=True)


def setup_argparser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description='Main starting point for running AlphaZero jobs')

    parser.add_argument('config', type=pathlib.Path,
                        help='Configuration file')
    parser.add_argument('--submit', action="store_true", default=False,
                        help='Submit the jobs to the HPC environment')

    return parser


if __name__ == "__main__":
    parser = setup_argparser()
    args = parser.parse_args()

    run(args.config, submit=args.submit)
