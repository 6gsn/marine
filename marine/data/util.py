from logging import getLogger
from pathlib import Path

from hydra.utils import to_absolute_path
from joblib import load
from torch.utils.data import DataLoader

from .dataset import AccentDataset
from .pad import Padsequence

logger = getLogger(__name__)


def load_dataset(config, phases=None):
    dataloader = {}

    data_dir = Path(to_absolute_path(config.data.data_dir))

    if phases is None:
        phases = ["train", "val", "test"]
    elif not isinstance(phases, list):
        raise TypeError(f"Unvailable values: {phases}")

    for phase in phases:
        is_train = phase == "train"
        targets = ["features", "labels", "ids"]
        data = {}

        for target in targets:
            data_path = data_dir / phase / f"{target}.pkl"
            data[target] = load(data_path)

        dataset = AccentDataset(data)

        if logger is not None:
            logger.info(f"{phase} data size : {len(dataset):,}")

        if is_train:
            shuffle = True
        else:
            shuffle = False

        dataloader[phase] = DataLoader(
            dataset,
            batch_size=config.data.batch_size,
            shuffle=shuffle,
            collate_fn=Padsequence(
                input_keys=config.data.input_keys,
                input_length_key=config.data.input_length_key,
                output_keys=config.data.output_keys,
                num_classes=config.data.output_sizes,
            ),
            num_workers=config.data.num_workers,
        )

    return dataloader
