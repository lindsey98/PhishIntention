import argparse

from detectron2.config import get_cfg
from detectron2_1.adv import DAGAttacker
from detectron2 import model_zoo
from detectron2_1.datasets import WebMapper


def main(args):
    print("Preparing config file...")
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg_path)
    cfg.MODEL.WEIGHTS = args.weights_path

    print("Initializing attacker...")
    # Using custom DatasetMapper
    attacker = DAGAttacker(cfg, mapper=WebMapper)

    print("Start the attack...")
    coco_instances_results = attacker.run_DAG(
        results_save_path=args.results_save_path, vis_save_dir=args.vis_save_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg-path",
        required=True,
        help="Path to configuration file used to train the model",
    )
    parser.add_argument("--weights-path", required=True, help="Path to model weights")
    parser.add_argument(
        "--results-save-path",
        required=True,
        help="Path to save the prediction results as a JSON file",
    )
    parser.add_argument(
        "--vis-save-dir",
        required=False,
        help="Directory to save visualized bbox prediction images",
    )

    args = parser.parse_args()

    main(args)
