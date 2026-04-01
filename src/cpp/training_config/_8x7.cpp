static azb::TrainConfig make_8x7_config() {
    azb::TrainConfig cfg;

    // ── Board ────────────────────────────────────────────────
    cfg.rows = 7;
    cfg.cols = 8;

    // ── Network ──────────────────────────────────────────────
    // Larger board needs deeper network to capture long-range
    // chain interactions across the board.
    cfg.hidden_size    = 512;
    cfg.num_res_blocks = 10;
    cfg.dropout        = 0.15;

    // ── MCTS ─────────────────────────────────────────────────
    cfg.c_puct            = 3.0f;   // high: near-random policy needs wide search
    cfg.dirichlet_alpha   = 0.12f;  // ~10 / avg_legal_moves (≈97 edges early)
    cfg.dirichlet_epsilon = 0.35f;  // more noise weight early on large board
    cfg.fpu_reduction     = 0.0f;   // keep unvisited nodes attractive in bootstrap

    // ── Global defaults ───────────────────────────────────────
    cfg.mcts_sims         = 600;
    cfg.episodes_per_iter = 100;
    cfg.batch_size        = 512;    
    cfg.epochs            = 12;
    cfg.learning_rate     = 0.001;
    cfg.num_iters         = 110;
    cfg.temp_threshold    = 20;
    cfg.temp_explore      = 1.0f;
    cfg.temp_exploit      = 0.5f;

    cfg.buffer_capacity   = 150000;
    cfg.buffer_grow       = 1000;

    // ── Inference ────────────────────────────────────────────
    cfg.num_workers           = 64;
    cfg.max_inference_batch   = 128;

    // ── Checkpoints ──────────────────────────────────────────
    cfg.keep_checkpoints = 5;        
    cfg.model_name       = "alphazero_8x7";
    cfg.model_dir        = "../models/8x7";

    // ── Phases ───────────────────────────────────────────────
    // NOTE: After each phase, manually update cfg.c_puct,
    // cfg.dirichlet_alpha, cfg.dirichlet_epsilon, and
    // cfg.fpu_reduction to match the phase schedule below.
    // Phase struct only carries the 8 fields shown.
    cfg.phases = {
        // Phase 1 — Bootstrap (iters 1–20)
        // c_puct=3.0, dir_alpha=0.12, dir_eps=0.35, fpu=0.0
        // Near-random policy: maximize board diversity in buffer.
        // temp_threshold=20 keeps 20% of the game exploratory.
        { "Bootstrap",    20, 200,  100, 10, 0.003f,  20, 1.0f, 0.50f },

        // Phase 2 — Region recognition (iters 21–40)
        // c_puct=2.5, dir_alpha=0.12, dir_eps=0.30, fpu=0.0
        // Network learns to partition board into chain regions.
        // Keep temp_explore=1.0 for diverse opening strategies.
        { "RegionRecog",  20, 400,  100, 12, 0.001f,  15, 1.0f, 0.35f },

        // Phase 3 — Chain tactics (iters 41–65)
        // c_puct=2.0, dir_alpha=0.12, dir_eps=0.25, fpu=0.2
        // 600 sims: multi-step chain forcing sequences visible.
        // fpu_reduction=0.2 begins pushing search deeper.
        { "ChainTactics", 25, 600,  100, 15, 0.0005f, 12, 0.80f, 0.20f },

        // Phase 4 — Parity & strategy (iters 66–90)
        // c_puct=1.5, dir_alpha=0.12, dir_eps=0.15, fpu=0.35
        // 800 sims to plan across 8+ simultaneous chains.
        // fpu=0.35 forces deep search into chain forcing lines.
        { "Parity",       25, 800,  80,  18, 0.0002f, 8,  0.60f, 0.08f },

        // Phase 5 — Mastery (iters 91–110)
        // c_puct=1.2, dir_alpha=0.10, dir_eps=0.08, fpu=0.35
        // Near-frozen LR, near-greedy, minimal noise.
        // 1000 sims for highest-quality self-play data.
        { "Mastery",      20, 1000, 60,  20, 0.00005f,5,  0.40f, 0.02f },
    };

    return cfg;
}