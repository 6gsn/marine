from logging import getLogger

from hydra.utils import instantiate

logger = getLogger(__name__)


def init_model(tasks, config, feature_set, device, is_train=False):
    if is_train:
        criterions = {}
        optimizer = None
        scheduler = None

    # setting shared layers
    # Embedding
    embedding_kwargs = {"feature_set": feature_set}
    embedding = instantiate(config.model.embedding, **embedding_kwargs)

    # Encoder
    encooder_input_size = sum(
        [config.model.embedding.embeding_sizes[key] for key in config.data.input_keys]
    )

    encoder_output_size = config.model.encoder.param.hidden_size * 2
    encoder_kwargs = {"input_size": encooder_input_size}

    encoders = {}
    for task in tasks:
        if (
            config.model.encoder.shared_with[task]
            and config.model.encoder.shared_with[task] in encoders.keys()
        ):
            logger.info(
                f"{task} has shared encoder with {config.model.encoder.shared_with[task]}"
            )
            encoders[task] = encoders[config.model.encoder.shared_with[task]]
        else:
            encoder = instantiate(config.model.encoder.param, **encoder_kwargs)
            encoders[task] = encoder

    # Decoder
    decoder_input_sizes = [encoder_output_size] * len(tasks)

    decoder_kwargs = {
        task: {
            "input_size": decoder_input_size,
            "output_size": config.data.output_sizes[task],
        }
        for task, decoder_input_size in zip(tasks, decoder_input_sizes)
    }

    # init decoders with criterion
    decoders = {}
    for task in tasks:
        # init embedding for
        if config.model.decoder[task]["prev_task_embedding_label_list"]:
            assert config.model.decoder[task]["prev_task_embedding_label_list"] == list(
                config.model.decoder[task]["prev_task_embedding_size"].keys()
            ), "Not matched embedding setting for previous tasks: {} != {}".format(
                config.model.decoder[task]["prev_task_embedding_label_list"],
                config.model.decoder[task]["prev_task_embedding_size"].keys(),
            )

            decoder_kwargs[task]["prev_task_embedding_label_size"] = {
                task_label: decoder_kwargs[task_label]["output_size"]
                for task_label in config.model.decoder[task][
                    "prev_task_embedding_label_list"
                ]
                if task_label in tasks
            }

        decoders[task] = instantiate(config.model.decoder[task], **decoder_kwargs[task])

        if is_train:
            if config.criterions[task]._target_ == "marine.criterions.LogLikelhood":
                criterion_kwargs = {"log_likehood_func": decoders[task].crf}
            else:
                criterion_kwargs = {}

            criterion = instantiate(config.criterions[task], **criterion_kwargs).to(
                device
            )
            criterions[task] = criterion

    # Init base model
    model_kwarwgs = {"embedding": embedding, "encoders": encoders, "decoders": decoders}

    model = instantiate(config.model.base, **model_kwarwgs).to(device)

    logger.debug(f"model has initialized\n{model}")

    if is_train:
        optimizer_kwargs = {"params": model.parameters()}
        optimizer = instantiate(config.optim.optimizer, **optimizer_kwargs)

        scheduler_kwargs = {"optimizer": optimizer}
        scheduler = instantiate(config.optim.scheduler, **scheduler_kwargs)

        return model, criterions, optimizer, scheduler
    else:
        return model
