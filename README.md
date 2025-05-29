## Project Description

This implementation is a research prototype that combines cutting-edge Vision-Language Models (VLMs), Large Language Models (LLMs) and a Tree-of-Thoughts planner to enable zero-shot object navigation and interaction in the RoboTHOR simulator. It begins by building and refining real-time RGB-D semantic maps, then uses a frontier-based exploration strategy (guided by VLM outputs) to identify and prioritize unexplored regions. A deterministic policy module translates the VLM’s high-level reasoning into precise navigation and manipulation commands, while a natural-language interface allows users to issue free-form commands like “find the red mug in the kitchen” and have the robot execute them on unseen objects.

Benchmarked on both simulated and real-world tasks, this project delivers high navigation accuracy, robust object-detection precision and reliable command-interpretation. Its zero-shot learning capability requires no object-specific training, and its two-stage mapping pipeline and dynamic updates ensure the robot maintains an accurate, up-to-date understanding of its environment.

## Setup & Installation

# Base Environment

Create the conda environment:
```sh
conda env create -n VLTNet -f environment.yml
```
Activate the environment:
```sh
conda activate VLTNet
```

Run the following to ensure that torch is properly installed.
```
python scripts/test_torch_download.py
```


# GLIP setup and model download
Setup GLIP with the following command
```
cd GLIP
python setup.py build develop --user
mkdir MODEL
wget https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_large_model.pth
```
make sure to verify that the downloaded pth file is around 6.9GB

# GPT-3.5 setup
visit the tree_of_thoughts.py and replace your openai key at line 17

# Pasture Benchmark Setup

To download the Pasture THOR binaries (~4GB) see below. This is a required step to run evaluations. Navigate to the repo root directory (`cow/`) and run the following:
```sh
wget https://cow.cs.columbia.edu/downloads/pasture_builds.tar.gz
```
```sh
tar -xvf pasture_builds.tar.gz
```
This should create a folder called `pasture_builds/`

To download episode targets and ground truth for evaluation, run the following:
```sh
wget https://cow.cs.columbia.edu/downloads/datasets.tar.gz
```
```sh
tar -xvf datasets.tar.gz
```
This should create a folder called `datasets/`

Additionally, THOR rendering requires that `Xorg` processes are running on all GPUs. If processes are not already running, run the following:
```
sudo python scripts/startx.py
```

# Evaluation on Pasture and RoboTHOR

Note: it is recommended to run evaluations in a `tmux` session as they are long running jobs.

For Pasture and RoboTHOR, to evaluate VLTNET, run:

```
python VLTNet_runner.py -a src.models.VLTNet -n 1 --reasoning both --cfg glip_config.yaml --visulize
```

to evaluate ESC, run:
```
python glip_runner.py -a src.models.GLIP_FBE_PSL -n 1 --reasoning both --cfg glip_config.yaml --visulize
```
Note: this automatically evaluates all Pasture splits and RoboTHOR. If the script is stopped, it will resume where it left off. If you want to re-evaluate from scratch, remove the results subfolder associated with the agent being evaluated in `results/`.

# Habitat MP3D Benchmark Setup

Not setup yet, please refer to CoW for further adaptations

# Helpful Pointers

Evaluation is often long running. Each time an evaluation episode completes, a `json` with information about the trajectory is stored in the `results/` folder. For example, for the default agent on the Pasture uncommon object split: `results/longtail_longtail_fbe_owl-b32-openai-center/*.json`. This allows for printing the completed evaluations, e.g.,

```
python success_agg.py --result-dir results/VLTNet_robothor_regular/
```
or

```
python success_agg.py --result-dir results/GLIP_robothor_regular/
```

# Visualization on Pasture
To visualize both an egocentric trajectory view and a top-down path as in the teaser gif above, run:

```
python path_visualization.py --out-dir viz/ --thor-floor FloorPlan_Val3_5 --result-json media/media_data/FloorPlan_Val3_5_GingerbreadHouse_1.json --thor-build pasture_builds/thor_build_longtail/longtail.x86_64
```

The script outputs 1) egocentric pngs for each view, 2) an mp4 for the egocentric feed, 3) top-down pngs for each pose, 4) an mp4 for the top-down feed. Video creation utilizes `ffmpeg` by making `os.system(...)` calls.

Note: flag arguments should be swapped accordinly for the floor plan and trajectory you wish to visualize. This script provides functionality to visualize RoboTHOR or Pasture evaluations.

Script based on code sourced from [here](https://github.com/allenai/cordial-sync/issues/5).


