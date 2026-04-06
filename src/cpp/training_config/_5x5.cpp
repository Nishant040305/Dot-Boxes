static azb::TrainConfig make_5x5_config() {
    azb::TrainConfig cfg;

    // ════════════════════════════════════════════════════════
    //  BOARD
    // ════════════════════════════════════════════════════════
    cfg.rows = 5;
    cfg.cols = 5;

    // ════════════════════════════════════════════════════════
    //  NETWORK
    // ════════════════════════════════════════════════════════
    cfg.hidden_size    = 256;
    cfg.num_res_blocks = 8;
    cfg.dropout        = 0.08f;

    // ════════════════════════════════════════════════════════
    //  MCTS
    // ════════════════════════════════════════════════════════
    // c_puct raised to 1.6 — matching 4x3. 5x5 has a much larger
    // branching factor so the tree NEEDS more exploration pressure
    // to find forcing sequences. 1.35 was starving the tree.
    cfg.c_puct            = 1.6f;

    // dirichlet_alpha lowered to 0.12 — 5x5 has more moves so
    // each individual move needs less noise weight to stay diverse.
    cfg.dirichlet_alpha   = 0.12f;
    cfg.dirichlet_epsilon = 0.25f;

    // fpu_reduction matches 4x3 — penalise unvisited nodes enough
    // that the tree commits to promising lines rather than scattering.
    cfg.fpu_reduction     = 0.20f;
    cfg.use_dag           = false;

    // ════════════════════════════════════════════════════════
    //  SELF-PLAY  (global defaults, overridden per phase)
    // ════════════════════════════════════════════════════════
    cfg.mcts_sims         = 400;
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
    cfg.buffer_capacity = 200000;

    // buffer_grow raised to 5000 — matches 4x3 philosophy.
    // 2000 was the silent killer: the buffer was filling with
    // stale early-phase games and drowning out new learning.
    // 5000 flushes stale data faster so each phase's lessons
    // actually dominate the training batch.
    cfg.buffer_grow     = 5000;

    // ════════════════════════════════════════════════════════
    //  PARALLELISM
    // ════════════════════════════════════════════════════════
    cfg.num_workers         = 16;
    cfg.max_inference_batch = 128;  // raised from 32 — matches 4x3

    // ════════════════════════════════════════════════════════
    //  CHECKPOINTS
    // ════════════════════════════════════════════════════════
    cfg.keep_checkpoints = 3;
    cfg.model_name       = "alphazero_5x5";
    cfg.model_dir        = "../models/_5x5";

    // __ Evaluation Matrix ____________________________________
    cfg.value_eval = azb::ValueEval::kScoreDiffTanh;

    // ════════════════════════════════════════════════════════
    //  PHASES  —  4 phases, mirroring 4x3 structure exactly
    //
    //  Root cause of previous failure: 7 phases spread the learning
    //  too thin. The 4x3 hammers each phase hard (epochs up to 30,
    //  temp_exploit down to 0.01) before moving on. The 5x5 config
    //  was babying the model across too many gentle transitions.
    //
    //  Key fixes applied from 4x3 analysis:
    //    1. epochs ramps 12 → 18 → 25 → 30  (was stuck at 15)
    //    2. temp_exploit reaches 0.01 in Mastery  (was 0.04–0.08)
    //    3. c_puct = 1.6  (was 1.35, tree was under-exploring)
    //    4. buffer_grow = 5000  (was 2000, stale data dominated)
    //    5. max_inference_batch = 128  (was 32, unnecessary bottleneck)
    //    6. LR 0.003 start — aggressive initial learning on loaded
    //       weights catches the network up fast without instability
    //
    //  Game counts:
    //    Phase 1  Bootstrap   :  25 × 300 =   7,500
    //    Phase 2  ChainAware  :  70 × 250 =  17,500
    //    Phase 3  DeepSearch  :  65 × 200 =  13,000
    //    Phase 4  Mastery     :  40 × 150 =   6,000
    //                                       ────────
    //                           TOTAL      =  44,000 games
    //
    //  NOTE: 44k games is intentionally modest. The 4x3 needed
    //  only 20k games to beat alpha-beta depth-3. 5x5 is harder
    //  but the fix is not MORE games — it is the right training
    //  signal per game. Run this full pass first; if ChainAware
    //  plateau is visible in loss curves, double that phase only.
    //
    //  Time estimate (i7-13700H):
    //    Phase 1 (200 sims, warm-up)    :  ~15 min
    //    Phase 2 (400 sims, chain burn) :  ~60 min
    //    Phase 3 (700 sims, deep)       :  ~80 min
    //    Phase 4 (900 sims, mastery)    :  ~55 min
    //                                     ─────────
    //                         TOTAL     =  ~3.5 hrs
    // ════════════════════════════════════════════════════════
    cfg.phases = {

        // ── Phase 1: Bootstrap ───────────────────────────────
        // Resume from existing weights. Short, high-LR warm-up.
        // 200 sims is intentionally low — we want cheap games that
        // refresh the replay buffer with current-policy data quickly.
        // epochs=12 is enough to absorb the new games without
        // overwriting the square-making knowledge already present.
        // temp_exploit=0.2 — loose, lets the model re-explore
        // positions it already knows. Not trying to learn here,
        // just re-anchoring the resumed weights.
        {
            /*name*/              "Bootstrap",
            /*iterations*/        25,
            /*mcts_sims*/         200,
            /*episodes_per_iter*/ 300,
            /*epochs*/            12,
            /*lr*/                0.003f,
            /*temp_threshold*/    8,
            /*temp_explore*/      1.0f,
            /*temp_exploit*/      0.20f
        },

        // ── Phase 2: ChainAware ──────────────────────────────
        // The make-or-break phase. Directly mirrors 4x3's Phase 2.
        // This is where chain tactics MUST be burned in:
        //   — do not give away long chains
        //   — sacrifice short chains to gain long ones
        //   — recognise forcing sequences 3–4 moves ahead
        //
        // epochs=18: same as 4x3. The network needs many gradient
        // steps per batch of games to internalise subtle chain value.
        // One epoch per game batch is nowhere near enough — the
        // signal from a forced sacrifice is rare and weak unless
        // the network sees it repeatedly per update cycle.
        //
        // temp_exploit=0.05: much more aggressive than before.
        // The model must COMMIT to its chain assessment, not sample.
        // At 0.15+ the policy stays soft enough that the model
        // "accidentally" gives away chains and never gets a clean
        // training signal that it was wrong.
        //
        // 400 sims at 5x5 sees ~4-move forcing sequences cleanly.
        {
            /*name*/              "ChainAware",
            /*iterations*/        70,
            /*mcts_sims*/         400,
            /*episodes_per_iter*/ 250,
            /*epochs*/            18,
            /*lr*/                0.001f,
            /*temp_threshold*/    6,
            /*temp_explore*/      0.80f,
            /*temp_exploit*/      0.05f
        },

        // ── Phase 3: DeepSearch ──────────────────────────────
        // Directly mirrors 4x3's Phase 3. DAG is now dense —
        // abort rate very low, every sim is effective.
        // 700 sims reaches deep enough to evaluate full endgame
        // on 5x5 (avg ~20 moves, 700 sims covers the critical
        // branching points with sufficient visit counts).
        //
        // epochs=25: same as 4x3. At this stage the value head
        // is doing fine-grained chain counting. More gradient
        // steps per game batch means the value estimates converge
        // to accurate chain counts rather than noisy approximations.
        //
        // temp_exploit=0.03: near-deterministic. Policy must be
        // sharp enough that the model plays the same move twice
        // from the same position. Anything above 0.05 allows
        // enough sampling variance to produce inconsistent play
        // that alpha-beta can exploit.
        {
            /*name*/              "DeepSearch",
            /*iterations*/        65,
            /*mcts_sims*/         700,
            /*episodes_per_iter*/ 200,
            /*epochs*/            25,
            /*lr*/                0.0005f,
            /*temp_threshold*/    5,
            /*temp_explore*/      0.60f,
            /*temp_exploit*/      0.03f
        },

        // ── Phase 4: Mastery ─────────────────────────────────
        // Directly mirrors 4x3's Phase 4. Policy sharpening only.
        // 900 sims is the ceiling for 5x5 on i7-13700H before
        // marginal returns collapse — tree is near-saturated and
        // extra sims refine visit distributions rather than
        // discovering new lines.
        //
        // epochs=30: same as 4x3. At near-zero LR, each epoch
        // is essentially free in terms of catastrophic forgetting
        // risk but provides another pass over the current game
        // batch, further sharpening the policy toward the unique
        // best move in each position.
        //
        // temp_exploit=0.01: matches 4x3 exactly. This is
        // functionally greedy (the argmax move gets ~99% of
        // probability mass at this temperature). AlphaZero is
        // now playing like alpha-beta — always the best known move.
        //
        // LR=0.0001: matches 4x3. Just enough gradient to keep
        // the weights from drifting, not enough to change them.
        {
            /*name*/              "Mastery",
            /*iterations*/        40,
            /*mcts_sims*/         900,
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
