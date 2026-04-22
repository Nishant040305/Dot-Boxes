static azb::TrainConfig make_5x5_config() {
    azb::TrainConfig cfg;

    // ════════════════════════════════════════════════════════
    //  BOARD
    // ════════════════════════════════════════════════════════
    cfg.rows = 5;
    cfg.cols = 5;

    // ════════════════════════════════════════════════════════
    //  NETWORK  —  scaled up for 5x5 complexity
    //
    //  5x5 state space is ~2^60 vs 4x3's ~2^31.  The network
    //  must reason about chains across 25 boxes (vs 12), which
    //  requires both wider layers (384 vs 128) and deeper
    //  residual tower (10 vs 6) to propagate chain information
    //  across the full board.
    // ════════════════════════════════════════════════════════
    cfg.hidden_size    = 384;
    cfg.num_res_blocks = 10;
    cfg.dropout        = 0.06f;

    // ════════════════════════════════════════════════════════
    //  MCTS
    // ════════════════════════════════════════════════════════
    //  c_puct = 1.6: 5x5 has ~60 legal moves early on. Higher
    //  exploration pressure is essential to find forcing sequences
    //  in the large branching factor.
    cfg.c_puct            = 1.6f;

    //  dirichlet_alpha = 0.10: lower than 4x3's 0.15 because
    //  each move needs less noise when there are more moves.
    //  0.10 * ~60 moves ≈ 6.0 total noise mass, matching 4x3's
    //  0.15 * ~30 ≈ 4.5 (slightly higher to encourage exploration
    //  in the larger space).
    cfg.dirichlet_alpha   = 0.10f;
    cfg.dirichlet_epsilon = 0.25f;

    cfg.fpu_reduction     = 0.20f;
    cfg.use_dag           = false;

    // ════════════════════════════════════════════════════════
    //  SELF-PLAY  (global defaults, overridden per phase)
    // ════════════════════════════════════════════════════════
    cfg.mcts_sims         = 600;
    cfg.episodes_per_iter = 200;
    cfg.num_iters         = 200;

    // ════════════════════════════════════════════════════════
    //  TRAINING
    // ════════════════════════════════════════════════════════
    cfg.batch_size    = 256;
    cfg.epochs        = 12;
    cfg.learning_rate = 0.001f;

    // ════════════════════════════════════════════════════════
    //  TEMPERATURE  (global defaults)
    // ════════════════════════════════════════════════════════
    cfg.temp_threshold = 8;
    cfg.temp_explore   = 1.0f;
    cfg.temp_exploit   = 0.3f;

    // ════════════════════════════════════════════════════════
    //  REPLAY BUFFER
    // ════════════════════════════════════════════════════════
    //  With 8× symmetry augmentation (D₄ group on square board),
    //  each game produces ~60 moves × 8 = 480 training positions.
    //  DIAGNOSTIC: augmentation disabled to test if it's corrupting
    //  training data.  Without 8× aug, each game produces ~60 samples.
    //  Buffer sized for ~500 games of data.
    cfg.use_augmentation = true;
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
    cfg.keep_checkpoints = 0;
    cfg.model_name       = "alphazero_5x5";
    cfg.model_dir        = "../models/_5x5";

    // ════════════════════════════════════════════════════════
    //  VALUE EVALUATION  —  kScoreDiffScaled
    //
    //  This is the CRITICAL fix.  kScoreDiffTanh divides by
    //  total_boxes before tanh, which crushes the training
    //  signal on 5x5 (values cluster in [-0.2, +0.2]).
    //
    //  kScoreDiffScaled uses tanh(0.5 * raw_score_diff):
    //    1-box lead → 0.46   (strong, discriminative)
    //    2-box lead → 0.76   (clearly winning)
    //    3-box lead → 0.91   (decisive)
    //
    //  Board-size invariant — same signal on 4x3, 5x5, or 7x8.
    // ════════════════════════════════════════════════════════
    cfg.value_eval = azb::ValueEval::kScoreDiffScaled;

    // ════════════════════════════════════════════════════════
    //  PHASE  —  Single Phase
    //
    //  200k games total (1000 iterations × 200 games).
    //  Estimated time (i7-13700H, post-optimization): ~8-12 hrs.
    // ════════════════════════════════════════════════════════
    cfg.phases = {

        // ── Single Phase: 200k games ─────────────────────────────
        {
            /*name*/              "Main",
            /*iterations*/        1000,
            /*mcts_sims*/         800,
            /*episodes_per_iter*/ 200,
            /*epochs*/            3,
            /*lr*/                0.001f,
            /*temp_threshold*/    8,
            /*temp_explore*/      1.0f,
            /*temp_exploit*/      0.30f,
            /*capture_boost*/     0.0f
        },

    };

    return cfg;
}
