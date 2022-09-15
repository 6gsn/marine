import json
import random
import time
from pathlib import Path
from shutil import copyfile

import hydra
import torch
from hydra.utils import to_absolute_path
from marine.bin.test import test_model
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
    init_seed,
    log_scores,
    pack_inputs,
    pack_outputs,
    pad_incomplete_accent_logits,
    plot_batch_attention,
)
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logger = None


def save_checkpoint(
    config, checkpoint_dir, model, optimizer, scheduler, epoch, is_best
):
    if model is None:
        return

    # save config file
    if epoch == 0:
        if not config.train.save_vocab_path:
            config.model.vocab_path = None

        config_path = checkpoint_dir / "config.yaml"
        logger.info(f"Save config: {config_path}")

        OmegaConf.save(config, config_path)

    is_interval = epoch > 0 and (epoch + 1) % config.train.checkpoint_interval == 0

    # save checkpoint when epoch is interval
    if is_interval or is_best:
        states = {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "optimizer": optimizer.state_dict()
            if config.train.save_optimizer_state and optimizer is not None
            else None,
            "scheduler": scheduler.state_dict()
            if config.train.save_optimizer_state and scheduler is not None
            else None,
        }

        if is_interval:
            interval_checkpoint_path = checkpoint_dir / f"epoch_{epoch:05d}.pth"
            lastest_path = checkpoint_dir / "latest.pth"
            logger.info(f"Save interval checkpoint: {interval_checkpoint_path}")
        else:
            interval_checkpoint_path = None
            lastest_path = None

        if is_best:
            best_checkpoint_path = checkpoint_dir / "best.pth"
            logger.info(f"Save best checkpoint: {best_checkpoint_path}")
        else:
            best_checkpoint_path = None

        if is_interval and is_best:
            torch.save(states, interval_checkpoint_path)
            copyfile(interval_checkpoint_path, best_checkpoint_path)
            copyfile(interval_checkpoint_path, lastest_path)

        elif is_interval:
            torch.save(states, interval_checkpoint_path)
            copyfile(interval_checkpoint_path, lastest_path)

        else:
            torch.save(states, best_checkpoint_path)


def train_model(
    model,
    criterions,
    optimizer,
    scheduler,
    dataloader,
    tasks,
    tensorboard_writer,
    checkpoint_dir,
    config,
    feature_set,
    num_epochs=10,
    device="cpu",
):
    phases = ["train", "val"]
    min_val_loss = [100.0] * len(tasks)
    fig_logging_targets = random.choices(range(config.data.batch_size), k=10)

    if "accent_status" in tasks:
        has_att_based_model = isinstance(
            model.decoders["accent_status"], AttentionBasedLSTMDecoder
        )
    else:
        has_att_based_model = False

    for epoch in range(num_epochs):
        logger.info("-" * 10)
        logger.info("Epoch {}/{}".format(epoch, num_epochs - 1))
        logger.info("-" * 10)

        since = time.time()

        for phase in phases:
            is_train = phase == "train"

            if is_train:
                model.train()
            else:
                model.eval()

            # for logging
            running_loss = {task: 0.0 for task in tasks}
            metrics = MultiTaskMetrics(
                phase,
                config.data.output_sizes,
                accent_represent_mode=config.data.represent_mode,
                require_ap_level_f1_score=has_att_based_model,
                device=device,
            )

            for batch_index, (inputs, outputs, _, script_ids) in enumerate(
                tqdm(dataloader[phase], desc=f"{phase}: ", leave=False)
            ):
                # pack inputs to device
                inputs = pack_inputs(inputs, config.data.input_keys, device)
                outputs = pack_outputs(outputs, device)

                # total loss of the tasks on a single batch
                batch_loss = 0.0
                prev_decoder_output = {}

                optimizer.zero_grad()

                for index, task in enumerate(tasks):
                    output, output_mask = outputs[task]["label"], outputs[task]["mask"]

                    # predict
                    with torch.set_grad_enabled(is_train):
                        decoder_outputs = model(task, **inputs)

                        if isinstance(model.decoders[task], CRFDecoder):
                            linear_logits, crf_logits = decoder_outputs
                            loss = criterions[task](linear_logits, output, output_mask)
                            logits = crf_logits
                        elif isinstance(model.decoders[task], LinearDecoder):
                            logits = decoder_outputs
                            loss = criterions[task](logits, output, output_mask)
                        elif isinstance(
                            model.decoders[task], AttentionBasedLSTMDecoder
                        ):
                            logits, attentions, ap_lengths = decoder_outputs
                            # plot attention when first batch on eval
                            if (
                                not is_train
                                and tensorboard_writer
                                and batch_index == 0
                                and task == "accent_status"
                            ):
                                plot_batch_attention(
                                    inputs,
                                    logits,
                                    ap_lengths,
                                    attentions,
                                    feature_set,
                                    plot_targets=fig_logging_targets,
                                    tensorboard_writer=tensorboard_writer,
                                    phase=phase,
                                    epoch=epoch,
                                    script_ids=script_ids,
                                )

                            # resize logits
                            logits = pad_incomplete_accent_logits(logits, output_mask)
                            loss = criterions[task](logits, output, output_mask)

                        batch_loss += loss
                        running_loss[task] += loss

                        # logits: (B, T, dim) -> (B, T)
                        predicts = torch.argmax(logits, dim=2)

                        # Update metrics
                        # for ap-based seq
                        if task == "accent_status" and has_att_based_model:
                            metrics.update(
                                task,
                                predicts,
                                output,
                                output_mask,
                                predicted_accent_phrase_boundaries=inputs[
                                    "prev_decoder_outputs"
                                ]["accent_phrase_boundary"],
                                target_accent_phrase_boundaries=outputs[
                                    "accent_phrase_boundary"
                                ]["label"],
                                mora_seq_masks=outputs["accent_phrase_boundary"][
                                    "mask"
                                ],
                            )

                        # for mora-based seq
                        else:
                            metrics.update(task, predicts, output, output_mask)

                        # cascade prev outputs
                        if index < len(tasks) - 1:
                            prev_decoder_output[task] = predicts

                            if is_train:
                                # Use golden label as teacher forcing
                                inputs["prev_decoder_outputs"][task] = output

                                if (
                                    index + 1 < len(tasks)
                                    and tasks[index + 1] == "accent_status"
                                    and has_att_based_model
                                ):
                                    inputs["decoder_targets"] = outputs[
                                        "accent_status"
                                    ]["label"]
                            else:
                                _output = prev_decoder_output[task]
                                inputs["prev_decoder_outputs"][task] = _output
                                inputs["decoder_targets"] = None

                if is_train:
                    batch_loss.backward()
                    optimizer.step()

            epoch_loss = {
                task: running_loss[task] / len(dataloader[phase]) for task in tasks
            }

            # Logging scores
            log_scores(
                phase,
                epoch,
                tasks,
                metrics,
                loss=epoch_loss,
                tensorboard_writer=tensorboard_writer,
            )

            if not is_train:
                current_val_losses = list(epoch_loss.values())

                # update schedule with validation loss
                scheduler.step(sum(current_val_losses))
                epoch_lr = optimizer.param_groups[0]["lr"]
                tensorboard_writer.add_scalar(f"{phase}_learning_rate", epoch_lr, epoch)

                # Save checkpoints
                is_best = [loss.item() for loss in current_val_losses] < min_val_loss
                min_val_loss = current_val_losses

                save_checkpoint(
                    config,
                    checkpoint_dir,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    is_best,
                )

        time_elapsed = time.time() - since
        logger.info(f"complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")


@hydra.main(config_path="conf/train", config_name="config")
def my_app(config: DictConfig) -> None:
    global logger
    logger = getLogger(config.train.verbose)
    logger.info(OmegaConf.to_yaml(config))

    init_seed(config.train.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = config.train.num_epochs

    dataloader = load_dataset(config)
    tasks = config.data.output_keys

    checkpoint_dir = (
        Path(to_absolute_path(config.train.out_dir)) / config.train.model_name
    )

    if not checkpoint_dir.exists():
        logger.info(f"Created checkpoint dir at {checkpoint_dir}")
        checkpoint_dir.mkdir(parents=True)

    # Setup tensorboard summary writer
    if config.train.tensorboard_event_path is None:
        tensorboard_event_path = f"tensorboard/{checkpoint_dir.name}"
    else:
        tensorboard_event_path = config.train.tensorboard_event_path
    tensorboard_event_path = Path(to_absolute_path(tensorboard_event_path))

    logger.info(f"TensorBoard event log path: {tensorboard_event_path}")
    tensorboard_writer = SummaryWriter(tensorboard_event_path)

    # Setup test result logging
    if config.train.test_log_dir:
        log_dir = Path(to_absolute_path(config.train.test_log_dir))
    else:
        log_dir = Path(to_absolute_path("logs")) / checkpoint_dir.name

    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    log_path = log_dir / f"{checkpoint_dir.name}_test_log.json"

    # Init feature set
    feature_set = FeatureSet(
        Path(to_absolute_path(config.model.vocab_path)),
        feature_table_key=config.data.feature_table_key,
        feature_keys=config.data.input_keys,
    )

    # Init model
    model, criterions, optimizer, scheduler = init_model(
        tasks, config, feature_set, device, is_train=True
    )

    # Train
    train_model(
        model,
        criterions,
        optimizer,
        scheduler,
        dataloader,
        tasks,
        tensorboard_writer,
        checkpoint_dir,
        config,
        feature_set,
        num_epochs=num_epochs,
        device=device,
    )

    # Init test model
    _test_model = init_model(tasks, config, feature_set, device)
    logs = test_model(
        _test_model,
        checkpoint_dir,
        config.train.test_checkpoint_filename,
        tasks,
        dataloader,
        config,
        feature_set,
        tensorboard_writer=tensorboard_writer,
        device=device,
        logger=logger,
    )

    if config.train.save_test_log:
        # save test result log
        with open(log_path, "w") as file:
            json.dump(
                logs,
                file,
                ensure_ascii=False,
                indent=4,
                separators=(",", ": "),
            )

    tensorboard_writer.close()


def entry():
    my_app()


if __name__ == "__main__":
    my_app()
