#!/usr/bin/env python

import sys
from copy import deepcopy
from pathlib import Path

from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor.rename_processor import rename_stats
import lerobot.scripts.lerobot_train as lerobot_train


DATA_ROOT = Path(
    "/home/cluster/smolvla_robomme_memory/data/robomme_full"
)

OUTPUT_DIR = Path(
    "/home/cluster/smolvla_robomme_memory/outputs/smolvla_robomme_full_baseline_30k"
)

RENAME = {
    "image": "observation.images.image",
    "wrist_image": "observation.images.wrist_image",
    "state": "observation.state",
    "actions": "action",
}


class RuntimeMetaAdapter:
    """
    In-memory metadata view.
    Does NOT modify info.json / stats.json on disk.
    """

    def __init__(self, original_meta):
        self._original = original_meta

        self.info = deepcopy(original_meta.info)

        old_features = deepcopy(original_meta.features)
        new_features = {}

        for key, value in old_features.items():
            new_features[RENAME.get(key, key)] = value

        self.info["features"] = new_features

        # Make normalization stats use canonical names too.
        self.stats = rename_stats(
            original_meta.stats,
            RENAME,
        )

    @property
    def features(self):
        return self.info["features"]

    @property
    def fps(self):
        return self._original.fps

    @property
    def camera_keys(self):
        return [
            key
            for key, ft in self.features.items()
            if ft["dtype"] in ("video", "image")
        ]

    def __getattr__(self, name):
        return getattr(self._original, name)


class RoboMMESmolVLADataset(Dataset):
    """
    Read-only runtime adapter:
        image       -> observation.images.image
        wrist_image -> observation.images.wrist_image
        state       -> observation.state
        actions     -> action

    Original RoboMME files remain untouched.
    """

    def __init__(self):
        self.dataset = LeRobotDataset(
            "lerobot/robomme",
            root=DATA_ROOT,
            delta_timestamps={
                # RoboMME = 10 FPS.
                # SmolVLA chunk_size = 50.
                "actions": [i / 10 for i in range(50)],
            },
        )

        self.meta = RuntimeMetaAdapter(self.dataset.meta)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx]

        out = dict(x)

        out["observation.images.image"] = out.pop("image")
        out["observation.images.wrist_image"] = out.pop("wrist_image")
        out["observation.state"] = out.pop("state")
        out["action"] = out.pop("actions")

        if "actions_is_pad" in out:
            out["action_is_pad"] = out.pop("actions_is_pad")

        return out

    def __getattr__(self, name):
        # Delegate num_frames, num_episodes, fps, etc.
        return getattr(self.dataset, name)


def make_robomme_dataset(cfg):
    print("\n========================================")
    print("RoboMME -> SmolVLA runtime adapter")
    print("Original dataset will NOT be modified.")
    print("========================================\n")

    return RoboMMESmolVLADataset()


def main():
    # Hard safety guard: NEVER overwrite an existing experiment.
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"\nREFUSING TO OVERWRITE EXISTING OUTPUT:\n"
            f"{OUTPUT_DIR}\n\n"
            f"Use a new output directory instead."
        )

    # Extra safety: this script is specifically the smoketest4 run.
    requested_output = None

    for arg in sys.argv[1:]:
        if arg.startswith("--output_dir="):
            requested_output = Path(arg.split("=", 1)[1]).resolve()

    if requested_output is not None:
        if requested_output != OUTPUT_DIR.resolve():
            raise ValueError(
                f"This smoke-test script only permits:\n"
                f"{OUTPUT_DIR}\n"
                f"but received:\n"
                f"{requested_output}"
            )

    # Replace ONLY the dataset factory in memory.
    # /home/cluster/lerobot is NOT modified.
    lerobot_train.make_dataset = make_robomme_dataset

    lerobot_train.main()


if __name__ == "__main__":
    main()
