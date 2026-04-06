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
    //  120k buffer ≈ 250 games of data, keeping it fresh.
    //  Old 200k buffer was drowning late-phase data in stale
    //  early-phase games.
    cfg.buffer_capacity = 120000;
    cfg.buffer_grow     = 3000;

    // ════════════════════════════════════════════════════════
    //  PARALLELISM
    // ════════════════════════════════════════════════════════
    cfg.num_workers         = 16;
    cfg.max_inference_batch = 128;

    // ════════════════════════════════════════════════════════
    //  CHECKPOINTS
    // ════════════════════════════════════════════════════════
    cfg.keep_checkpoints = 3;
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
    //  PHASES  —  4 phases, mirroring 4x3 structure
    //
    //  Key changes from previous config:
    //    1. kScoreDiffScaled — 2-5× stronger value signal
    //    2. Full D₄ augmentation — 8× data (was 3×)
    //    3. value_eval passed to MCTS agent — consistent
    //       terminal evaluation (was mismatched!)
    //    4. MCTS sims scaled with action space:
    //       DeepSearch 1000 sims / 60 actions ≈ 17 visits/action
    //       (comparable to 4x3's 600/31 ≈ 19)
    //    5. Network 384h × 10 blocks — adequate for 25-box
    //       chain reasoning
    //
    //  Game counts:
    //    Phase 1  Bootstrap   :  30 × 300 =   9,000
    //    Phase 2  ChainAware  :  80 × 250 =  20,000
    //    Phase 3  DeepSearch  :  70 × 200 =  14,000
    //    Phase 4  Mastery     :  50 × 150 =   7,500
    //                                       ────────
    //                           TOTAL      =  50,500 games
    //                           × 60 moves × 8 augment = ~24M positions
    //
    //  Time estimate (i7-13700H):
    //    Phase 1 (250 sims)     :  ~20 min
    //    Phase 2 (600 sims)     :  ~80 min
    //    Phase 3 (1000 sims)    :  ~100 min
    //    Phase 4 (1400 sims)    :  ~75 min
    //                              ─────────
    //                  TOTAL    =  ~4.5 hrs
    // ════════════════════════════════════════════════════════
    cfg.phases = {

        // ── Phase 1: Bootstrap ───────────────────────────────
        //  Warm-up with moderate sims. 8× augmentation means
        //  even 300 games produce 300×60×8 = 144k positions.
        //  LR 0.003 for fast initial learning.
        {
            /*name*/              "Bootstrap",
            /*iterations*/        30,
            /*mcts_sims*/         250,
            /*episodes_per_iter*/ 300,
            /*epochs*/            12,
            /*lr*/                0.003f,
            /*temp_threshold*/    8,
            /*temp_explore*/      1.0f,
            /*temp_exploit*/      0.20f
        },

        // ── Phase 2: ChainAware ──────────────────────────────
        //  600 sims ≈ 10 visits/action — enough to see 3-4 move
        //  forcing sequences at 5x5 branching factor.
        //  epochs=18 to burn in chain tactics.
        {
            /*name*/              "ChainAware",
            /*iterations*/        80,
            /*mcts_sims*/         600,
            /*episodes_per_iter*/ 250,
            /*epochs*/            18,
            /*lr*/                0.001f,
            /*temp_threshold*/    6,
            /*temp_explore*/      0.80f,
            /*temp_exploit*/      0.05f
        },

        // ── Phase 3: DeepSearch ──────────────────────────────
        //  1000 sims ≈ 17 visits/action — comparable to 4x3's
        //  600/31 ≈ 19 visits/action. Deep enough for endgame
        //  chain counting on 5x5.
        {
            /*name*/              "DeepSearch",
            /*iterations*/        70,
            /*mcts_sims*/         1000,
            /*episodes_per_iter*/ 200,
            /*epochs*/            25,
            /*lr*/                0.0005f,
            /*temp_threshold*/    5,
            /*temp_explore*/      0.60f,
            /*temp_exploit*/      0.03f
        },

        // ── Phase 4: Mastery ─────────────────────────────────
        //  1400 sims ≈ 23 visits/action — comparable to 4x3's
        //  800/31 ≈ 26 visits/action. Near-saturated tree at
        //  near-zero temperature for policy sharpening.
        {
            /*name*/              "Mastery",
            /*iterations*/        50,
            /*mcts_sims*/         1400,
            /*episodes_per_iter*/ 150,
            /*epochs*/            30,
            /*lr*/                0.0001f,
            /*temp_threshold*/    4,
            /*temp_explore*/      0.40f,
            /*temp_exploit*/      0.01f
        },
    };

    return cfg;
}
