static azb::TrainConfig make_4x3_config() {
    azb::TrainConfig cfg;

    // ── Board ────────────────────────────────────────────────
    cfg.rows = 4;
    cfg.cols = 3;

    // ── Network ──────────────────────────────────────────────
    cfg.hidden_size    = 128;
    cfg.num_res_blocks = 6;     // slightly deeper
    cfg.dropout        = 0.1f;  // less regularization (small state space)

    // ── MCTS ─────────────────────────────────────────────────
    cfg.c_puct            = 1.6f;   
    cfg.dirichlet_alpha   = 0.15f;  
    cfg.dirichlet_epsilon = 0.20f;
    cfg.fpu_reduction     = 0.2f;   

    // ── Global defaults ──────────────────────────────────────
    cfg.mcts_sims         = 400;
    cfg.episodes_per_iter = 200;
    cfg.batch_size        = 256;
    cfg.epochs            = 12;
    cfg.learning_rate     = 0.001;
    cfg.num_iters         = 120;      // key change (~200k games total)
    cfg.temp_threshold    = 6;
    cfg.temp_explore      = 1.0f;
    cfg.temp_exploit      = 0.3f;

    // ── Replay buffer ────────────────────────────────────────
    cfg.buffer_capacity   = 50000;
    cfg.buffer_grow       = 5000;

    // ── Inference ────────────────────────────────────────────
    cfg.num_workers           = 16;
    cfg.max_inference_batch   = 128;

    // ── Checkpoints ──────────────────────────────────────────
    cfg.keep_checkpoints = 5;
    cfg.model_name       = "alphazero_4x3";
    cfg.model_dir        = "../models/_4x3";

    // ── Phases ───────────────────────────────────────────────
    cfg.phases = {

        // Phase 1 — Stabilize basics
        { "Bootstrap", 20, 200, 300, 12, 0.003f, 6, 1.0f, 0.2f },

        // Phase 2 — Learn chains properly
        { "ChainAware", 50, 300, 250, 18, 0.001f, 5, 0.8f, 0.05f },

        // Phase 3 — Deep search pressure
        { "DeepSearch", 30, 600, 200, 25, 0.0005f, 4, 0.6f, 0.03f },

        // Phase 4 — Near-perfect policy
        { "Mastery", 20, 800, 150, 30, 0.0001f, 3, 0.4f, 0.01f },
    };

    return cfg;
}