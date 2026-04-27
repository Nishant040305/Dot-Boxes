static azb::TrainConfig make_5x5_cnn_kaggle_config() {
    azb::TrainConfig cfg;

    // ════════════════════════════════════════════════════════
    //  BOARD
    // ════════════════════════════════════════════════════════
    cfg.rows = 5;
    cfg.cols = 5;

    // ════════════════════════════════════════════════════════
    //  NETWORK  —  CNN architecture (same as local config)
    //
    //  Uses 2D convolutional residual blocks instead of
    //  fully-connected layers.  The board is encoded as an
    //  11-channel spatial tensor of shape (6, 6) for 5×5.
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
    //
    //  Kaggle T4 GPU:  NN inference is ~10× faster than CPU.
    //  Bottleneck shifts to CPU-side MCTS tree operations.
    //  With only 4 CPU cores we use 4 workers, but can afford
    //  higher sims since GPU inference is essentially free.
    // ════════════════════════════════════════════════════════
    cfg.mcts_sims         = 1600;
    cfg.episodes_per_iter = 100;   // fewer games per iter (4 workers vs 16)
    cfg.num_iters         = 200;

    // ════════════════════════════════════════════════════════
    //  TRAINING
    //
    //  GPU training is much faster, so we can afford more
    //  epochs per iteration without a time penalty.
    // ════════════════════════════════════════════════════════
    cfg.batch_size    = 256;
    cfg.epochs        = 12;
    cfg.learning_rate = 0.0003f;

    cfg.value_loss_weight  = 1.0f;
    cfg.policy_loss_weight = 1.0f;
    cfg.grad_clip_norm     = 5.0f;

    // Entropy regularization: prevents policy head collapse
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
    // ════════════════════════════════════════════════════════
    cfg.use_augmentation = false;
    cfg.buffer_capacity = 60000;
    cfg.buffer_grow     = 0;

    // ════════════════════════════════════════════════════════
    //  PARALLELISM — tuned for Kaggle T4 (4 CPU cores)
    //
    //  - 4 workers  (1 per core, no over-subscription)
    //  - Larger inference batch since GPU can handle it and
    //    we want to amortize kernel launch overhead
    // ════════════════════════════════════════════════════════
    cfg.num_workers         = 4;
    cfg.max_inference_batch = 64;    // 4 workers can't fill 128

    // ════════════════════════════════════════════════════════
    //  CHECKPOINTS — save frequently for 12h session safety
    // ════════════════════════════════════════════════════════
    cfg.keep_checkpoints = 3;
    cfg.model_name       = "alphazero_5x5_cnn";
    cfg.model_dir        = "/kaggle/working/models_5x5_cnn";

    // ════════════════════════════════════════════════════════
    //  VALUE EVALUATION
    // ════════════════════════════════════════════════════════
    cfg.value_eval = azb::ValueEval::kScoreDiffScaled;

    // ════════════════════════════════════════════════════════
    //  PHASES  —  optimised for ~12h Kaggle session
    //
    //  With 4 workers + GPU inference, each iteration takes
    //  roughly 3–5 min (100 games × 1600 sims).  That gives
    //  us ~150–200 iterations in 12 hours.
    //
    //  Phase 1: Warm-up — standard sims, build buffer
    //  Phase 2: Deep search — boost sims, fewer episodes
    //           to squeeze more quality per game
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
            /*iterations*/        250,
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
