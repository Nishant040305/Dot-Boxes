static azb::TrainConfig make_4x3_config() {
    azb::TrainConfig cfg;

    // ── Board ────────────────────────────────────────────────
    cfg.rows = 3;
    cfg.cols = 4;

    // ── Network ──────────────────────────────────────────────
    cfg.hidden_size    = 128;
    cfg.num_res_blocks = 4;
    cfg.dropout        = 0.1;

    // ── MCTS ─────────────────────────────────────────────────
    cfg.c_puct            = 2.0f;   
    cfg.dirichlet_alpha   = 0.4f;   
    cfg.dirichlet_epsilon = 0.30f;
    cfg.fpu_reduction     = 0.0f;   

    // ── Global defaults (overridden by phases) ────────────────
    cfg.mcts_sims         = 400;
    cfg.episodes_per_iter = 80;
    cfg.batch_size        = 128;
    cfg.epochs            = 8;
    cfg.learning_rate     = 0.001;
    cfg.num_iters         = 30;     
    cfg.temp_threshold    = 6;
    cfg.temp_explore      = 1.0f;
    cfg.temp_exploit      = 0.4f;

    // ── Replay buffer ────────────────────────────────────────
    cfg.buffer_capacity   = 20000;
    cfg.buffer_grow       = 300;

    // ── Inference ────────────────────────────────────────────
    cfg.num_workers           = 32;
    cfg.max_inference_batch   = 32;

    // ── Checkpoints ──────────────────────────────────────────
    cfg.keep_checkpoints = 5;
    cfg.model_name       = "alphazero_4x3";
    cfg.model_dir        = "../models/4x3";
    cfg.phases = {
        { "Bootstrap",   12,  150, 80,  8,  0.003f, 6, 1.0f, 0.40f },
        { "ChainAware",  10,  300, 100, 10, 0.001f, 4, 0.8f, 0.15f },
        { "Mastery",     8,   400, 80,  12, 0.0003f,3, 0.5f, 0.05f },
    };

    return cfg;
}