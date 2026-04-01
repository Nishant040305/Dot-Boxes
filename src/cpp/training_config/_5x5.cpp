static azb::TrainConfig make_5x5_config() {
    azb::TrainConfig cfg;

    cfg.rows = 5;
    cfg.cols = 5;

    cfg.hidden_size    = 256;
    cfg.num_res_blocks = 6;
    cfg.dropout        = 0.1;

    cfg.c_puct            = 2.5f;   
    cfg.dirichlet_alpha   = 0.30f;  
    cfg.dirichlet_epsilon = 0.30f;
    cfg.fpu_reduction     = 0.0f;

    cfg.mcts_sims         = 400;
    cfg.episodes_per_iter = 100;
    cfg.batch_size        = 256;
    cfg.epochs            = 10;
    cfg.learning_rate     = 0.001;
    cfg.num_iters         = 55;
    cfg.temp_threshold    = 10;
    cfg.temp_explore      = 1.0f;
    cfg.temp_exploit      = 0.3f;

    cfg.buffer_capacity   = 50000;
    cfg.buffer_grow       = 500;

    cfg.num_workers           = 64;
    cfg.max_inference_batch   = 64;

    cfg.keep_checkpoints = 5;
    cfg.model_name       = "alphazero_5x5";
    cfg.model_dir        = "../models/5x5";

    cfg.phases = {
        { "Bootstrap",    15, 200, 100, 10, 0.002f,  10, 1.0f, 0.30f },
        { "Tactical",     15, 400, 100, 12, 0.001f,  8,  0.85f,0.20f },
        { "ChainParity",  15, 600, 80,  15, 0.0005f, 6,  0.70f,0.10f },
        { "Mastery",      10, 800, 60,  20, 0.0001f, 4,  0.50f,0.03f },
    };

    return cfg;
}