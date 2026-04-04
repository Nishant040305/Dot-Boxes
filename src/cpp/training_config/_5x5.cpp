static azb::TrainConfig make_5x5_config() {
    azb::TrainConfig cfg;

    cfg.rows = 5;
    cfg.cols = 5;

    // --- Network ---
    cfg.hidden_size    = 256;
    cfg.num_res_blocks = 8;     // slightly deeper
    cfg.dropout        = 0.1;

    // --- MCTS ---
    cfg.c_puct            = 2.25f;   // slightly reduced (more stable)
    cfg.dirichlet_alpha   = 0.30f;
    cfg.dirichlet_epsilon = 0.25f;   // less noise later
    cfg.fpu_reduction     = 0.25f;   // helps early search stability

    // --- Self-play ---
    cfg.mcts_sims         = 400;     // will scale in phases
    cfg.episodes_per_iter = 100;
    cfg.num_iters         = 400;     // 40k games total

    // --- Training ---
    cfg.batch_size     = 256;
    cfg.epochs         = 10;
    cfg.learning_rate  = 0.001;

    // --- Temperature ---
    cfg.temp_threshold = 12;
    cfg.temp_explore   = 1.0f;
    cfg.temp_exploit   = 0.25f;

    // --- Replay buffer ---
    cfg.buffer_capacity = 80000;   // larger for longer training
    cfg.buffer_grow     = 500;

    // --- Parallelism ---
    cfg.num_workers         = 64;
    cfg.max_inference_batch = 64;

    // --- Checkpoints ---
    cfg.keep_checkpoints = 5;
    cfg.model_name       = "alphazero_5x5";
    cfg.model_dir        = "../models/_5x5";

    // --- Training Phases (sum = 400 iters) ---
    cfg.phases = {
        // name         iters sims eps  epochs lr       temp_th exp   exploit
        { "Bootstrap",   50,  200, 120, 8,     0.002f,  12,     1.0f, 0.30f },
        { "Expansion",   80,  400, 100, 10,    0.0015f, 10,     0.9f, 0.25f },
        { "Tactical",    100, 600, 80,  12,    0.001f,  8,      0.75f,0.20f },
        { "Refinement",  100, 800, 60,  15,    0.0005f, 6,      0.6f, 0.10f },
        { "Mastery",     70,  1000,50,  20,    0.0001f, 4,      0.5f, 0.05f },
    };

    return cfg;
}