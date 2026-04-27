static azb::TrainConfig make_5x5_cnn_config() {
    azb::TrainConfig cfg;

    // ════════════════════════════════════════════════════════
    //  BOARD
    // ════════════════════════════════════════════════════════
    cfg.rows = 5;
    cfg.cols = 5;

    // ════════════════════════════════════════════════════════
    //  NETWORK  —  CNN architecture
    //
    //  Uses 2D convolutional residual blocks instead of
    //  fully-connected layers.  The board is encoded as an
    //  11-channel spatial tensor of shape (6, 6) for 5×5.
    //
    //  CNN advantages:
    //    - Local spatial patterns (edge adjacency) learned natively
    //    - Weight sharing across board positions
    //    - More parameter-efficient than FC for large boards
    // ════════════════════════════════════════════════════════
    cfg.use_cnn_net     = true;
    cfg.cnn_channels    = 128;     // conv feature channels
    cfg.num_res_blocks  = 10;      // conv residual blocks
    cfg.dropout         = 0.06f;

    // FC params not used by CNN, but set for compatibility
    cfg.hidden_size     = 384;

    // ════════════════════════════════════════════════════════
    //  MCTS
    // ════════════════════════════════════════════════════════
    cfg.c_puct            = 1.6f;
    cfg.dirichlet_alpha   = 0.10f;
    cfg.dirichlet_epsilon = 0.25f;
    cfg.fpu_reduction     = 0.20f;
    cfg.use_dag           = false;

    // ════════════════════════════════════════════════════════
    //  SELF-PLAY
    // ════════════════════════════════════════════════════════
    cfg.mcts_sims         = 1600;
    cfg.episodes_per_iter = 200;
    cfg.num_iters         = 200;

    // ════════════════════════════════════════════════════════
    //  TRAINING
    // ════════════════════════════════════════════════════════
    cfg.batch_size    = 256;
    cfg.epochs        = 12;
    cfg.learning_rate = 0.0003f;

    cfg.value_loss_weight  = 1.0f;
    cfg.policy_loss_weight = 1.0f;
    cfg.grad_clip_norm     = 5.0f;

    // Entropy regularization: prevents the policy from staying near-uniform
    // over ~30 legal moves (the same plateau that stalled the FC model).
    cfg.entropy_coeff      = 0.01f;

    // ════════════════════════════════════════════════════════
    //  TEMPERATURE
    // ════════════════════════════════════════════════════════
    cfg.temp_threshold = 12;
    cfg.temp_explore   = 0.6f;
    cfg.temp_exploit   = 0.10f;

    // ════════════════════════════════════════════════════════
    //  POLICY TARGET SHARPENING
    // ════════════════════════════════════════════════════════
    cfg.policy_target_temp = 0.1f;

    // ════════════════════════════════════════════════════════
    //  REPLAY BUFFER
    //
    //  NOTE: Symmetry augmentation is disabled for CNN mode
    //  (spatial tensors need different augmentation logic).
    //  Buffer size is still large for diversity.
    // ════════════════════════════════════════════════════════
    cfg.use_augmentation = false;
    cfg.buffer_capacity = 60000;
    cfg.buffer_grow     = 0;

    // ════════════════════════════════════════════════════════
    //  PARALLELISM
    // ════════════════════════════════════════════════════════
    cfg.num_workers         = 16;
    cfg.max_inference_batch = 128;

    // ════════════════════════════════════════════════════════
    //  CHECKPOINTS
    // ════════════════════════════════════════════════════════
    cfg.keep_checkpoints = 5;  // retain checkpoints for NaN rollback
    cfg.model_name       = "alphazero_5x5_cnn";
    cfg.model_dir        = "../models/_5x5_cnn";

    // ════════════════════════════════════════════════════════
    //  VALUE EVALUATION
    // ════════════════════════════════════════════════════════
    cfg.value_eval = azb::ValueEval::kScoreDiffScaled;

    // ════════════════════════════════════════════════════════
    //  PHASES
    // ════════════════════════════════════════════════════════
    cfg.phases = {
        {
            /*name*/              "p1",
            /*iterations*/        200,
            /*mcts_sims*/         1600,
            /*episodes_per_iter*/ 200,
            /*epochs*/            4,
            /*lr*/                0.0003f,
            /*temp_threshold*/    12,
            /*temp_explore*/      0.6f,
            /*temp_exploit*/      0.10f,
            /*capture_boost*/     0.0f
        },
        {
            /*name*/              "p2",
            /*iterations*/        400,
            /*mcts_sims*/         3200,
            /*episodes_per_iter*/ 200,
            /*epochs*/            4,
            /*lr*/                0.0003f,
            /*temp_threshold*/    12,
            /*temp_explore*/      0.6f,
            /*temp_exploit*/      0.10f,
            /*capture_boost*/     0.0f
        },
    };

    return cfg;
}
