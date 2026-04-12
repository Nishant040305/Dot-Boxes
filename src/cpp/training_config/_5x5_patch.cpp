static azb::TrainConfig make_5x5_patch_config() {
    azb::TrainConfig cfg;

    // ════════════════════════════════════════════════════════
    //  BOARD
    // ════════════════════════════════════════════════════════
    cfg.rows = 5;
    cfg.cols = 5;

    // ════════════════════════════════════════════════════════
    //  PATCH-NET ARCHITECTURE
    //
    //  The core idea: instead of training a massive monolithic
    //  network for 5x5, we reuse a FROZEN pre-trained 3x3 model
    //  to extract local tactical features from overlapping patches,
    //  then train only a lightweight global aggregator.
    //
    //  Benefits:
    //    1. Local captures/chains are pre-learned from 3x3 training
    //    2. Only 4 residual blocks need gradient updates (vs 10)
    //    3. ~60% fewer trainable parameters
    //    4. Converges much faster — local tactics don't need re-discovery
    //
    //  Patch layout (3x3 patches on 5x5 board):
    //    9 overlapping patches at (r0,c0) = (0..2, 0..2)
    //    Each patch produces 24 action logits + 1 value signal
    //    These are scattered/averaged onto the 60-action space
    //    and fused with global features by the aggregator.
    // ════════════════════════════════════════════════════════
    cfg.use_patch_net = true;

    // Local model (FROZEN — pre-trained 3x3)
    cfg.patch_rows           = 3;
    cfg.patch_cols           = 3;
    cfg.local_hidden_size    = 128;   // must match the 3x3 model architecture
    cfg.local_num_res_blocks = 6;     // must match the 3x3 model architecture
    // Path is relative to the patch model directory (e.g., src/cpp/models/_5x5_patch).
    cfg.local_model_path     = "../_3x3/alphazero_3x3.pt";

    // Global aggregator (TRAINABLE)
    cfg.global_hidden_size    = 256;  // wider than default 192 for 5x5 complexity
    cfg.global_num_res_blocks = 6;    // deeper aggregator for chain reasoning

    // These are used by standard-model code paths only (kept for compat)
    cfg.hidden_size    = 384;
    cfg.num_res_blocks = 10;
    cfg.dropout        = 0.08;

    // ════════════════════════════════════════════════════════
    //  MCTS
    // ════════════════════════════════════════════════════════
    cfg.c_puct            = 1.6f;
    cfg.dirichlet_alpha   = 0.10f;
    cfg.dirichlet_epsilon = 0.25f;
    cfg.fpu_reduction     = 0.20f;
    cfg.use_dag           = false;

    // ════════════════════════════════════════════════════════
    //  SELF-PLAY (global defaults, overridden per phase)
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
    //  TEMPERATURE
    // ════════════════════════════════════════════════════════
    cfg.temp_threshold = 8;
    cfg.temp_explore   = 1.0f;
    cfg.temp_exploit   = 0.3f;

    // ════════════════════════════════════════════════════════
    //  REPLAY BUFFER
    //
    //  No symmetry augmentation for PatchNet (patch features
    //  would require separate permutation tables). Each game
    //  produces ~60 raw samples. Buffer sized for ~500 games.
    // ════════════════════════════════════════════════════════
    cfg.use_augmentation = false;
    cfg.buffer_capacity  = 50000;
    cfg.buffer_grow      = 2000;

    // ════════════════════════════════════════════════════════
    //  PARALLELISM
    // ════════════════════════════════════════════════════════
    cfg.num_workers         = 16;
    cfg.max_inference_batch = 128;

    // ════════════════════════════════════════════════════════
    //  CHECKPOINTS
    // ════════════════════════════════════════════════════════
    cfg.keep_checkpoints = 3;
    cfg.model_name       = "alphazero_5x5_patch";
    cfg.model_dir        = "../models/_5x5_patch";

    // ════════════════════════════════════════════════════════
    //  VALUE EVALUATION
    // ════════════════════════════════════════════════════════
    cfg.value_eval = azb::ValueEval::kScoreDiffScaled;

    // ════════════════════════════════════════════════════════
    //  PHASES
    //
    //  Since local tactics are pre-learned from the 3x3 model,
    //  we can skip the Bootstrap phase entirely and start directly
    //  with global chain reasoning.
    //
    //  Phase 1: Warmup — let the global aggregator learn to use
    //           local signals. Higher LR, more exploration.
    //  Phase 2: ChainAware — medium sims, the aggregator learns
    //           to coordinate multi-patch chain strategies.
    //  Phase 3: DeepSearch — high sims to polish endgame play.
    //  Phase 4: Mastery — near-greedy exploitation.
    //
    //  Total: ~35,000 games (vs ~50,000 for monolithic 5x5)
    //  Expected wall-time: ~2-3 hours (vs ~4.5 hours monolithic)
    // ════════════════════════════════════════════════════════
    cfg.phases = {
        // ── Phase 1: Warmup ──────────────────────────────────
        //  Global aggregator learns to trust local signals.
        //  High LR, moderate exploration.
        {
            /*name*/              "PatchWarmup",
            /*iterations*/        40,
            /*mcts_sims*/         400,
            /*episodes_per_iter*/ 200,
            /*epochs*/            4,
            /*lr*/                0.003f,
            /*temp_threshold*/    10,
            /*temp_explore*/      1.0f,
            /*temp_exploit*/      0.30f,
            /*capture_boost*/     0.0f
        },

        // ── Phase 2: ChainAware ──────────────────────────────
        //  Aggregator learns cross-patch chain coordination.
        {
            /*name*/              "ChainAware",
            /*iterations*/        60,
            /*mcts_sims*/         600,
            /*episodes_per_iter*/ 200,
            /*epochs*/            6,
            /*lr*/                0.001f,
            /*temp_threshold*/    8,
            /*temp_explore*/      0.9f,
            /*temp_exploit*/      0.15f,
            /*capture_boost*/     0.0f
        },

        // ── Phase 3: DeepSearch ──────────────────────────────
        //  Higher sims refine endgame / parity play.
        {
            /*name*/              "DeepSearch",
            /*iterations*/        50,
            /*mcts_sims*/         900,
            /*episodes_per_iter*/ 150,
            /*epochs*/            8,
            /*lr*/                0.0003f,
            /*temp_threshold*/    6,
            /*temp_explore*/      0.8f,
            /*temp_exploit*/      0.08f,
            /*capture_boost*/     0.0f
        },

        // ── Phase 4: Mastery ─────────────────────────────────
        //  Near-greedy exploitation, minimal noise.
        {
            /*name*/              "Mastery",
            /*iterations*/        30,
            /*mcts_sims*/         1200,
            /*episodes_per_iter*/ 100,
            /*epochs*/            10,
            /*lr*/                0.0001f,
            /*temp_threshold*/    4,
            /*temp_explore*/      0.6f,
            /*temp_exploit*/      0.02f,
            /*capture_boost*/     0.0f
        },
    };

    return cfg;
}
