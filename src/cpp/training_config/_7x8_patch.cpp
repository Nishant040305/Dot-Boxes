static azb::TrainConfig make_7x8_patch_config() {
    azb::TrainConfig cfg;

    // ── Board ────────────────────────────────────────────────
    cfg.rows = 7;
    cfg.cols = 8;

    // ════════════════════════════════════════════════════════
    //  PATCH-NET ARCHITECTURE
    //
    //  7x8 board = 56 boxes, 113 edges.
    //  Monolithic approach: 512h × 10 blocks, huge compute.
    //
    //  Patch approach: use a frozen 4×3 model (12-box patches)
    //  on overlapping regions, then train a global aggregator.
    //
    //  Patch layout (4×3 patches on 7×8):
    //    n_patches_r = 7 - 4 + 1 = 4
    //    n_patches_c = 8 - 3 + 1 = 6
    //    Total: 24 overlapping patches
    //
    //  Why 4×3 local model (not 3×3)?
    //    - 3×3 = 9 boxes per patch, too small for 7×8 scale.
    //    - 4×3 = 12 boxes — can see meaningful chain fragments.
    //    - 4×3 model already exists and is well-trained.
    //    - 24 patches gives dense coverage of the 56-box space.
    //
    //  Compute savings:
    //    Monolithic: 512h × 10 blocks ≈ 5.2M params, all need gradients
    //    PatchNet:   128h × 6 blocks (frozen) + 256h × 6 blocks (trainable)
    //               ≈ 0.4M frozen + 1.2M trainable = ~77% param reduction
    // ════════════════════════════════════════════════════════
    cfg.use_patch_net = true;

    // Local model (FROZEN — pre-trained 4×3)
    cfg.patch_rows           = 4;
    cfg.patch_cols           = 3;
    cfg.local_hidden_size    = 128;   // must match the 4x3 model architecture
    cfg.local_num_res_blocks = 6;     // must match the 4x3 model architecture
    // Path is relative to the patch model directory (e.g., src/cpp/models/_7x8_patch).
    cfg.local_model_path     = "../_4x3/alphazero_4x3.pt";

    // Global aggregator (TRAINABLE)
    cfg.global_hidden_size    = 256;
    cfg.global_num_res_blocks = 6;

    // Standard net fallback (kept for compat)
    cfg.hidden_size    = 512;
    cfg.num_res_blocks = 10;
    cfg.dropout        = 0.10;

    // ── MCTS ─────────────────────────────────────────────────
    cfg.c_puct            = 2.0f;
    cfg.dirichlet_alpha   = 0.08f;  // ~10 / avg_legal_moves (≈113)
    cfg.dirichlet_epsilon = 0.30f;
    cfg.fpu_reduction     = 0.15f;
    cfg.use_dag           = false;

    // ── Global defaults ──────────────────────────────────────
    cfg.mcts_sims         = 600;
    cfg.episodes_per_iter = 150;
    cfg.batch_size        = 256;
    cfg.epochs            = 12;
    cfg.learning_rate     = 0.001;
    cfg.num_iters         = 150;
    cfg.temp_threshold    = 15;
    cfg.temp_explore      = 1.0f;
    cfg.temp_exploit      = 0.3f;

    cfg.use_augmentation  = false;  // No symmetry aug for PatchNet
    cfg.buffer_capacity   = 100000;
    cfg.buffer_grow       = 2000;

    // ── Inference ────────────────────────────────────────────
    cfg.num_workers           = 12;
    cfg.max_inference_batch   = 64;

    // ── Checkpoints ──────────────────────────────────────────
    cfg.keep_checkpoints = 3;
    cfg.model_name       = "alphazero_7x8_patch";
    cfg.model_dir        = "../models/_7x8_patch";

    // ── Value Evaluation ─────────────────────────────────────
    cfg.value_eval = azb::ValueEval::kScoreDiffScaled;

    // ════════════════════════════════════════════════════════
    //  PHASES
    //
    //  Since local 4×3 tactics are pre-learned:
    //    Phase 1: Warmup — aggregator learns to use 24 patch signals
    //    Phase 2: Regional — cross-patch chain coordination
    //    Phase 3: Strategic — parity and deep chain planning
    //    Phase 4: Mastery — high sims, near-greedy exploitation
    //
    //  Total: ~25,000 games (vs ~55,000 monolithic)
    //  Expected wall-time: ~3-4 hours (vs ~8+ hours monolithic)
    // ════════════════════════════════════════════════════════
    cfg.phases = {
        // Phase 1: Warmup
        {
            /*name*/              "PatchWarmup",
            /*iterations*/        30,
            /*mcts_sims*/         300,
            /*episodes_per_iter*/ 250,
            /*epochs*/            4,
            /*lr*/                0.003f,
            /*temp_threshold*/    20,
            /*temp_explore*/      1.0f,
            /*temp_exploit*/      0.40f,
            /*capture_boost*/     0.0f
        },

        // Phase 2: Regional coordination
        {
            /*name*/              "Regional",
            /*iterations*/        40,
            /*mcts_sims*/         500,
            /*episodes_per_iter*/ 200,
            /*epochs*/            6,
            /*lr*/                0.001f,
            /*temp_threshold*/    15,
            /*temp_explore*/      0.9f,
            /*temp_exploit*/      0.25f,
            /*capture_boost*/     0.0f
        },

        // Phase 3: Strategic chain planning
        {
            /*name*/              "Strategic",
            /*iterations*/        40,
            /*mcts_sims*/         800,
            /*episodes_per_iter*/ 150,
            /*epochs*/            8,
            /*lr*/                0.0003f,
            /*temp_threshold*/    10,
            /*temp_explore*/      0.8f,
            /*temp_exploit*/      0.10f,
            /*capture_boost*/     0.0f
        },

        // Phase 4: Mastery
        {
            /*name*/              "Mastery",
            /*iterations*/        25,
            /*mcts_sims*/         1000,
            /*episodes_per_iter*/ 100,
            /*epochs*/            10,
            /*lr*/                0.0001f,
            /*temp_threshold*/    6,
            /*temp_explore*/      0.5f,
            /*temp_exploit*/      0.03f,
            /*capture_boost*/     0.0f
        },
    };

    return cfg;
}
