import argparse
import json
import random
import sys
from pathlib import Path

import torch
from marine.data.feature.feature_set import FeatureSet
from marine.data.util import load_dataset
from marine.logger import getLogger
from marine.models import (
    AttentionBasedLSTMDecoder,
    CRFDecoder,
    LinearDecoder,
    init_model,
)
from marine.utils.metrics import MultiTaskMetrics
from marine.utils.util import (
    convert_readable_labels,
    group_by_script_id,
    init_seed,
    log_scores,
    pack_inputs,
    pack_outputs,
    pad_incomplete_accent_logits,
    plot_batch_attention,
)
from omegaconf import OmegaConf
from tqdm import tqdm

logger = None


def get_parser():
    parser = argparse.ArgumentParser(
        description="Test model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoint_dir", type=Path, help="Directory for checkpoint to test"
    )
    parser.add_argument(
        "--out_dir", "-o", type=Path, default=None, help="Directory test log"
    )
    parser.add_argument(
        "--checkpoint_filename",
        "-f",
        type=str,
        default="latest.pth",
        help="Model's file name to test",
    )
    parser.add_argument(
        "--data_dir", "-d", type=Path, default=None, help="Directory of dataset to test"
    )
    parser.add_argument(
        "--vocab_path", "-b", type=Path, default=None, help="Path of vocab file"
    )
    parser.add_argument("--n_jobs", type=int, default=8, help="Number of jobs")
    parser.add_argument(
        "--accent_status_represent_mode",
        "-m",
        type=str,
        choices=["binary", "high_low"],
        default="binary",
        help="Representation mode for accent status label",
    )
    parser.add_argument(
        "--random_seed",
        "-r",
        type=int,
        default=12345,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        default=50,
        help="Logging level",
    )
    return parser


def test_model(
    model,
    checkpoint_dir,
    checkpoint_file,
    tasks,
    dataloader,
    config,
    feature_set,
    tensorboard_writer=None,
    logger=None,
    device="cpu",
):
    model_path = checkpoint_dir / checkpoint_file
    states = torch.load(model_path)

    phase = "test"
    fig_logging_targets = random.choices(range(config.data.batch_size), k=10)

    logger.info(f"Load checkpoint from {model_path} ({states['epoch']}th epoch)")
    model.load_state_dict(states["state_dict"])

    dataloader = dataloader[phase]

    if "accent_status" in tasks:
        has_att_based_model = isinstance(
            model.decoders["accent_status"], AttentionBasedLSTMDecoder
        )
    else:
        has_att_based_model = False

    # for logging
    metrics = MultiTaskMetrics(
        phase,
        config.data.output_sizes,
        accent_represent_mode=config.data.represent_mode,
        require_ap_level_f1_score=has_att_based_model,
        device=device,
    )

    total_logs = {task: {} for task in tasks}

    for batch_index, (inputs, outputs, _, script_ids) in enumerate(
        tqdm(dataloader, desc=f"{phase}: ", leave=False)
    ):
        # pack inputs to device
        inputs = pack_inputs(inputs, config.data.input_keys, device)
        outputs = pack_outputs(outputs, device)

        prev_decoder_output = {}

        for task_index, task in enumerate(tasks):
            output, output_mask = outputs[task]["label"], outputs[task]["mask"]
            decoder_outputs = model(task, **inputs)

            # predict
            if isinstance(model.decoders[task], CRFDecoder):
                _, crf_logits = decoder_outputs
                logits = crf_logits
            elif isinstance(model.decoders[task], LinearDecoder):
                logits = decoder_outputs
            elif isinstance(model.decoders[task], AttentionBasedLSTMDecoder):
                logits, attentions, ap_lengths = decoder_outputs

                # plot attention when first batch on test
                if tensorboard_writer and batch_index == 0 and task == "accent_status":
                    plot_batch_attention(
                        inputs,
                        logits,
                        ap_lengths,
                        attentions,
                        feature_set,
                        plot_targets=fig_logging_targets,
                        tensorboard_writer=tensorboard_writer,
                        phase=phase,
                        epoch=0,
                        script_ids=script_ids,
                    )

                # resize logits
                logits = pad_incomplete_accent_logits(logits, output_mask)

            # logits: (B, T, dim) -> (B, T)
            predicts = torch.argmax(logits, dim=2)

            # Log predicts
            total_logs[task].update(
                convert_readable_labels(predicts, output, output_mask, script_ids)
            )

            # Update metrics
            # for ap-based seq
            if task == "accent_status" and has_att_based_model:
                metrics.update(
                    task,
                    predicts,
                    output,
                    output_mask,
                    predicted_accent_phrase_boundaries=inputs["prev_decoder_outputs"][
                        "accent_phrase_boundary"
                    ],
                    target_accent_phrase_boundaries=outputs["accent_phrase_boundary"][
                        "label"
                    ],
                    mora_seq_masks=outputs["accent_phrase_boundary"]["mask"],
                )

            # for mora-based seq
            else:
                metrics.update(task, predicts, output, output_mask)

            if task_index < len(tasks) - 1:
                prev_decoder_output[task] = predicts
                inputs["prev_decoder_outputs"][task] = predicts

    # Group by script_id
    total_logs = group_by_script_id(total_logs)

    # Logging scores
    log_scores(
        phase=phase,
        epoch=0,
        tasks=tasks,
        metrics=metrics,
        logs=total_logs,
        loss=None,
        tensorboard_writer=tensorboard_writer,
    )

    return total_logs


def entry(argv=sys.argv):
    global logger
    args = get_parser().parse_args(argv[1:])
    logger = getLogger(args.verbose)
    logger.debug(f"Loaded parameters: {args}")

    init_seed(args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = args.checkpoint_dir

    if not checkpoint_dir.exists():
        raise FileNotFoundError("Checkpoint dir not found")

    checkpoint_config_path = checkpoint_dir / "config.yaml"

    if not checkpoint_config_path.exists():
        raise FileNotFoundError("config file not found")

    checkpoint_config = OmegaConf.load(checkpoint_config_path)

    logger.info("Loaded config")
    logger.info(checkpoint_config)

    if args.data_dir:
        checkpoint_config.data.data_dir = str(args.data_dir)

    checkpoint_config.data.num_workers = args.n_jobs
    checkpoint_config.data.represent_mode = args.accent_status_represent_mode

    dataloader = load_dataset(checkpoint_config, phases=["test"])
    tasks = checkpoint_config.data.output_keys

    if args.out_dir:
        log_dir = Path(args.out_dir)
    else:
        log_dir = Path("logs") / checkpoint_dir.name

    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    log_path = log_dir / f"{checkpoint_dir.name}_test_log.json"

    # init feature set
    if args.vocab_path:
        if args.vocab_path.exists():
            feature_set = FeatureSet(
                args.vocab_path,
                feature_table_key=checkpoint_config.data.feature_table_key,
                feature_keys=checkpoint_config.data.input_keys,
            )
        else:
            raise FileNotFoundError(f"Not found vocab file: {args.vocab_path}")
    else:
        raise FileNotFoundError(
            "Please specify vocab file path in args or config/model/*.yaml"
        )

    # Init test model
    model = init_model(tasks, checkpoint_config, feature_set, device)
    logs = test_model(
        model,
        checkpoint_dir,
        args.checkpoint_filename,
        tasks,
        dataloader,
        checkpoint_config,
        feature_set,
        logger=logger,
        device=device,
    )

    # save log
    with open(log_path, "w") as file:
        json.dump(logs, file, ensure_ascii=False, indent=4, separators=(",", ": "))


if __name__ == "__main__":
    sys.exit(entry())
