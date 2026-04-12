static azb::TrainConfig make_2x2_config() {
    azb::TrainConfig cfg;

    // ── Board ────────────────────────────────────────────────
    cfg.rows = 2;
    cfg.cols = 2;

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
    cfg.model_name       = "alphazero_2x2";
    cfg.model_dir        = "../models/_2x2";

    // __ Evaluation Matrix ____________________________________
    cfg.value_eval = azb::ValueEval::kScoreDiffTanh;
    
    // ── Phases ───────────────────────────────────────────────
    cfg.phases = {
        { "SinglePhase", 200, 800, 100, 2, 0.003f, 6, 1.0f, 0.2f, 0.0f },
    };

    return cfg;
}