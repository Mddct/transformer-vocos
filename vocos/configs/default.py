import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # dataset
    config.train_dataset_path = ''
    config.eval_dataset_path = ''

    # train
    # Per device batch size for training.
    config.per_device_batch_size = 32
    # Per device batch size for training.
    config.eval_per_device_batch_size = 32
    config.max_train_steps = 500_000
    config.num_eval_steps = 2_000
    config.checkpoint_every_steps = 10_000
    # Base learning rate.
    config.learning_rate = 0.0016
    # Linear learning rate warmup.
    config.warmup_steps = 1000
    # Decay factor for AdamW style weight decay.
    config.weight_decay = 0.1
    # Save a checkpoint every these number of steps.
    config.checkpoint_every_steps = 10_000
    # Frequency of eval during training, e.g. every 1_000 steps.
    config.eval_every_steps = 1_000
    # Use bfloat16 mixed precision training instead of float32.
    config.use_bfloat16 = True
    # Integer for PRNG random seed.
    config.seed = 2025

    # mel
    config.sample_rate = 24000
    config.hop_size = 480  # sample_rate // hop_size = 50 for flow
    # TODO(Mddct:) other info

    # model
    # TODO(Mddct): change later when trainer is done
    config.output_size = 256
    config.attention_heads = 4
    config.linear_units = 2048
    config.num_blocks = 6
    config.dropout_rate = 0.1
    config.positional_dropout_rate = 0.1
    config.attention_dropout_rate = 0.0
    config.input_layer = ""
    config.pos_enc_layer_type = ""
    config.normalize_before = True
    config.static_chunk_size = 0
    config.use_dynamic_chunk = False
    config.global_cmvn = None
    config.use_dynamic_left_chunk = False
    config.query_bias = True
    config.key_bias = True
    config.value_bias = True
    config.activation_type = "relu"
    config.gradient_checkpointing = False
    config.use_sdpa = False
    config.layer_norm_type = "rms_norm"
    config.norm_eps = 1e-5
    config.n_kv_head = None
    config.head_dim = None
    config.selfattention_layer_type = "selfattn"
    config.mlp_type = "position_wise_feed_forward"
    config.mlp_bias = True
    config.n_expert = 8
    config.n_expert_activated = 2
    config.right_context = 2  # 2*6*20 =240ms
    config.left_context = 15  # 300ms

    return config
