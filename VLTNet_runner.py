
import os
from robothor_challenge import RobothorChallenge
import argparse
import importlib
import json
import logging
import torch

from src.simulation.sim_enums import ClassTypes, EnvTypes, POSIBLE_CONFIGS
logging.getLogger().setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser(
        description="Inference script for RoboThor ObjectNav challenge.")

    parser.add_argument(
        "--agent", "-a",
        required=True,
        help="Relative module for agent definition.",
    )

    parser.add_argument(
        "--template", "-t",
        required=False,
        default="prompt_templates/imagenet_template.json",
        help="Prompt template json.",
    )

    parser.add_argument(
        "--cfg", "-c",
        default="config_cow.yaml",
        help="Filepath to challenge config.",
    )

    parser.add_argument(
        "--output", "-o",
        default="metrics.json.gz",
        help="Filepath to output results to.",
    )

    parser.add_argument(
        "--remap-class-json",
        action='store',
        type=str,
        required=False,
        default=''
    )

    parser.add_argument(
        "--nprocesses", "-n",
        default=1,
        type=int,
        help="Number of parallel processes used to compute inference.",
    )

    parser.add_argument(
        "--PSL_infer", default="one_hot", type=str, choices=["optim", "one_hot"]
    )
    parser.add_argument(
        "--reasoning", default="obj", type=str, choices=["both", "room", "obj"]
    )
    parser.add_argument(
        "--error_analysis", action='store_true'
    )
    parser.add_argument(
        "--visulize", action='store_true'
    )

    parser.add_argument(
        "--model-path", "-p",
        default=''
    )

    parser.add_argument(
        "--semantic",
        default=False,
        action='store_true'
    )

    parser.add_argument(
        "--seed", "-s",
        default=0,
        required=False
    )

    parser.add_argument(
        "-nfs",
        default=False,
        action='store_true'
    )

    parser.add_argument(
        "-exp",
        default=False,
        action='store_true'
    )

    parser.add_argument(
        "--depth",
        default=False,
        action='store_true'
    )


    args = parser.parse_args()

    if 'gt' in args.agent:
        assert args.semantic

    experiments = [
        (EnvTypes.LONGTAIL, ClassTypes.LONGTAIL),
        (EnvTypes.ROBOTHOR, ClassTypes.REGULAR),
        (EnvTypes.DUP, ClassTypes.APPEARENCE),
        (EnvTypes.DUP, ClassTypes.SPATIAL),
        (EnvTypes.REMOVE, ClassTypes.HIDDEN),
        (EnvTypes.NORMAL, ClassTypes.APPEARENCE),
        (EnvTypes.NORMAL, ClassTypes.SPATIAL),
        (EnvTypes.NORMAL, ClassTypes.HIDDEN),
    ]


    for env_type, class_type in experiments:

        if not os.path.exists('results'):
            os.mkdir('results')

        experiment_name = f'results/VLTNet_{env_type.name.lower()}_{class_type.name.lower()}'

        cache = set()
        if os.path.exists(experiment_name):
            for p in os.listdir(experiment_name):
                cache.add(p.split('.')[0])
        if ('robothor' in experiment_name and len(cache) == 1800):
            continue
        elif len(cache) == 360:
            continue

        fail_stop = not args.nfs
        run_exploration_split = args.exp

        agent = importlib.import_module(args.agent)
        agent_class, agent_kwargs, render_depth = None, None, None

        
        agent_class, agent_kwargs, render_depth = agent.build()



        parser.add_argument(
        "--EnvTypes",
        default=env_type,
        action='store_true'
        )
     
        parser.add_argument(
        "--ClassTypes",
        default=class_type,
        action='store_true'
        )
        args = parser.parse_args()


        agent_kwargs['args'] = args
        


        class_remap = None
        if class_type in [ClassTypes.SPATIAL, ClassTypes.APPEARENCE]:
            class_remap = {}
            raw_annotation = None
            with open('class_templates/spatial_appearence_map.json', 'r') as f:
                raw_annotation = json.load(f)
            for scene in raw_annotation:
                class_remap[scene] = {}
                for object in raw_annotation[scene]:
                    if class_type == ClassTypes.SPATIAL:
                        class_remap[scene][object] = raw_annotation[scene][object][0]
                    else:
                        class_remap[scene][object] = raw_annotation[scene][object][1]
        elif class_type == ClassTypes.HIDDEN:
            with open('class_templates/hidden_map.json', 'r') as f:
                class_remap = json.load(f)
        
        r = RobothorChallenge(
            args.cfg,
            agent_class,
            agent_kwargs,
            experiment_name,
            env_type,
            class_type,
            render_depth=render_depth,
            render_segmentation=args.semantic,
            class_remap=class_remap)

        challenge_metrics = {}
        dataset_dir = None

        if env_type == EnvTypes.ROBOTHOR:
            dataset_dir = 'datasets/robothor-objectnav'
        elif env_type == EnvTypes.LONGTAIL:
            dataset_dir = 'datasets/robothor-objectnav-longtail'
        elif env_type == EnvTypes.NORMAL:
            if class_type == ClassTypes.SPATIAL or class_type == ClassTypes.APPEARENCE:
                dataset_dir = 'datasets/robothor-objectnav-normal'
            else:
                # hidden case
                dataset_dir = 'datasets/robothor-objectnav-hidden'
        elif env_type == EnvTypes.DUP:
            dataset_dir = 'datasets/robothor-objectnav-dup'
        elif env_type == EnvTypes.REMOVE:
            dataset_dir = 'datasets/robothor-objectnav-hidden'

        assert dataset_dir is not None

        val_episodes, val_dataset = r.load_split(dataset_dir, "val")

        if run_exploration_split:
            refined_val_episodes = []
            subsampled_episodes = None
            with open('robothor_exploration_episode_keys.json', 'r') as f:
                subsampled_episodes = set(json.load(f))

            for e in val_episodes:
                if e['id'] in subsampled_episodes:
                    refined_val_episodes.append(e)

            val_episodes = refined_val_episodes

        refined_val_episodes = []
        for v in val_episodes:
            if v['id'] not in cache:
                refined_val_episodes.append(v)
        
        challenge_metrics["val"] = r.inference(
            refined_val_episodes,
            nprocesses=args.nprocesses,
            test=False
        )


if __name__ == "__main__":
    main()
