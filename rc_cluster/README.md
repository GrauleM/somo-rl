# SoMo-RL on an RC Cluster

To run SoMo-RL experiments on a research cluster using Singularity and Slurm (commonly used for research computing), we **(1)** build the Singularity image, **(2)** load the Singularity image onto the cluster, **(3)** set up SoMo-RL on the cluster, and **(4)** enqueue jobs using Slurm.

___

## Using Singularity

**Installation**

Install singularity on your local machine following the Quick Installation Steps [here](https://sylabs.io/guides/3.7/user-guide/quick_start.html). You'll need a Linux system to use it (VMs work fine). 

**Recipe File**

The Singularity `.recipe` file defines how the Singularity image should be built. The default recipe we use is `containers/somo_rl.recipe`. This recipe file defines the following instructions:

1. Bootstrap a Ubuntu 18.04 Docker container to run inside the Singularity container.
2. Copy the SoMo-RL requirements file from the local machine to the Singularity image.
3. Install Python 3.7, pip, and system requirements from SoMo-RL.

If you want to change the installations in the Singularity image, edit the recipe file.

**Building**

Building a Singularity image takes a `.recipe` file and produces a `.img` file. 

> *Important Note:* In order to properly build the SoMo-RL Singularity image, you'll need to clone the [SoMo framework](https://github.com/GrauleM/somo) into this repo as `containers/somo`.

To build, run the following on your local machine:

```
$ cd containers
$ sudo singularity build ../cluster/somo_rl.img somo_rl.recipe
```

You need root permissions in order to build the Singularity image, which is why we do so on our local machine and upload the built image to the cluster.

**Creating an Overlay**

In order to have read/write access to the Singularity container, you need to create an *overlay*. You can read more about Singularity overlays [here](https://sylabs.io/guides/3.7/user-guide/persistent_overlays.html).

Run the following on your local machine to set up an overlay:

```
$ cd ../cluster
$ mkdir -p overlay/upper overlay/work
$ dd if=/dev/zero of=overlay.img bs=1M count=500 && mkfs.ext3 -d overlay overlay.img
```

> *Note:* We reccomend running the image locally before taking the time to upload the large `.img` files to a remote machine. Enter an interactive Singularity shell to test out the image by running `singularity shell --overlay overlay.img somo_rl.img`.

___

## Setting up SoMo-RL on a Cluster

After preparing the Singularity image, it's pretty simple to run it on a research cluster.

**Copy Images to Cluster**

Use `scp` to copy files over SSH:

```
$ scp somo_rl.img mccarthy@login.rc.fas.harvard.edu:~/somo_rl.img
$ scp overlay.img mccarthy@login.rc.fas.harvard.edu:~/overlay.img
$ scp experiment.sh mccarthy@login.rc.fas.harvard.edu:~/experiment.sh
$ scp experiment.slurm mccarthy@login.rc.fas.harvard.edu:~/experiment.slurm
```

**Setup SoMo-RL on Cluster**

Follow these steps to configure directories properly:

```bash
# SSH onto cluster
$ ssh mccarthy@login.rc.fas.harvard.edu
# Make overlay directories
$ mkdir -p overlay/upper overlay/work
```

Running `ls` from your home directory should now give you:
```
experiment.sh  experiment.slurm   overlay	overlay.img  somo_rl.img
```

Continue by setting up home storage directory:

```bash
# Enter home storage directory (this is where data will be stored)
$ cd /n/holyscratch01/wood_lab/Users/mccarthy
# Make RL work directory
$ mkdir rl_work
# Clone SoMo-RL
$ git clone https://github.com/t7mccarthy/somo-rl.git rl_work/somo-rl
# Make or clone experiments data with something like
$ mkdir rl_work/somo-rl-experiments
# or
$ git clone https://github.com/t7mccarthy/somo-rl-experiments.git rl_work/somo-rl-experiments
# Create user settings file
$ touch rl_work/somo-rl/somo_rl/user_settings.py
```
Running `ls` from your home storage directory should now give you `rl_work`.

Running `ls` from `rl_work/` should now give you:
```
somo-rl  somo-rl-experiments
```
Now we just need to populate the user settings file. Record the absolute path of your home storage directory (you can find this by running `pwd`). Run `$ nano rl_work/somo-rl/user_settings.py` to edit the file, and write in something of the form:

```python
USER = 'Tom FASRC Cluster'
EXPERIMENT_ABS_PATH = '/n/holyscratch01/wood_lab/Users/mccarthy/rl_work/somogym-baseline-results/experiments'
```
In this example, `/n/holyscratch01/wood_lab/Users/mccarthy` is the absolute path to the home storage directory. Replace it with your own.

___

## Running Experiments on the Cluster

**Testing in an Interactive Environment**

Slurm is a service that handles job queuing for shared compute resources. To test out the system on the remote, you can enter an interactive Slurm session and enter a Singularity shell:
```bash
# Enter an interactive Slurm session
$ salloc -n 1 -c 4 -N 1 --mem 16000 -t 0-04:00 --partition serial_requeue
# Enter an interactive Singularity shell
$ singularity shell --overlay overlay.img somo_rl.img
```
You can then run SoMo-RL experiments and tests as you would on your local machine (i.e. `python3.7 policy_training_script.py -e cluster_experiments -g group_0 -r run_0`).

> *Note:* Use `python3.7` to run a python script rather than `python` or `python3`. Singularity does not support aliasing.

> *Note:* When using Singularity, make sure that `stable_baselines3` is imported earlier than `gym` and `matplotlib` in your Python3 files.

**Running Experiments**

An experiment consists of multiple runs, each of which has a run config. Make sure the experiment you want to run is all set up in your experiments directory.

Relevant files:
- `experiment.slurm` is a bash script that runs a training run in Singularity when submitted as a Slurm job. The comments at the top configure Slurm job settings; edit these in order to set up your job differently.
- `experiment.sh` is a bash script that submits training runs from a given experiment as Slurm jobs. It takes in experiment name as an argument and calls `experiment.slurm`.

Usage:

Call the `experiment.sh` bash script with the `-e` or `--exp_name` flag for the experiment name argument. For example:
```
. experiment.sh -e singularity_experiment
```
This will submit a Slurm job for each run in the experiment and update your experiments directory with output.

**Checking on Jobs**

Stdout and stderr are saved into `rl_work/slurm_logs/[EXPERIEMT NAME]/[RUN NAME]/` as `output.out` and `error.err`.

Useful Slurm commands:
- View all jobs: `$ squeue -u mccarthy`
- Cancel all jobs: `$ scancel -u mccarthy`
- Cancel one job: `$ scancel [JOB ID]`

___
## Acknoledgements

- The process followed here is based on Alexander Koenig's experimentation flow on the FAS RC Cluster.

<!-- ___
___

## Docker
___
We also provide a Dockerfile to run SoMo-RL. Setup is largely the same as the steps listed above using Singularity, but you'll have to configure the proper docker directory yourself following Docker's documentation.

**Setup:**

Follow these instructions to set up docker locally: https://docs.docker.com/engine/install/ubuntu/

To build, run `sudo docker build {path to docker directory}`

Cheat sheet with some helpful docker commands: https://dockerlabs.collabnix.com/docker/cheatsheet/ -->