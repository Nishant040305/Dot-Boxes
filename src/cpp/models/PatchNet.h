#pragma once
/// PatchNet — Hierarchical Patch-Based Network for Dots and Boxes.
///
/// Architecture (inspired by CNN local receptive fields):
///   1. Extract overlapping KxK patches from the NxN board
///   2. Run a FROZEN pre-trained KxK model on each patch
///   3. Aggregate local policy/value signals into a spatial map
///   4. A lightweight global network combines local signals with
///      global features to produce the final policy + value.
///
/// Benefits:
///   - Local tactical patterns (captures, avoid 3-edge) are pre-learned
///   - Only the global aggregator needs training → much less compute
///   - Naturally handles spatial structure without CNN weight sharing
///   - Scales to larger boards by reusing the small model
///
/// Example: 5×5 board with 3×3 patch model
///   - 9 overlapping patches at positions (0,0)..(2,2)
///   - Each patch produces 24 action logits + 1 value
///   - Local signals are scattered to 60 global actions (averaged overlap)
///   - Global network: 4 res blocks, 192 hidden → final policy + value

#include <torch/torch.h>
#include "AlphaZeroBitNet.h"

namespace azb {

// ─── Patch descriptor ────────────────────────────────────────────────

struct PatchDesc {
    int r0, c0;  // top-left box coordinate on the big board

    // Maps local action index → global action index
    // local_to_global[local_idx] = global_idx
    std::vector<int> local_to_global;
};

// ─── PatchNet Implementation ─────────────────────────────────────────

struct PatchNetImpl : torch::nn::Module {

    // Board geometry
    int big_rows, big_cols;
    int patch_rows, patch_cols;
    int n_patches_r, n_patches_c, n_patches;
    int global_action_size, local_action_size;
    int global_feat_size, local_feat_size;

    // Pre-trained local model (FROZEN — no gradient)
    AlphaZeroBitNet local_model{nullptr};

    // Global feature processor (lightweight residual tower)
    torch::nn::Linear input_fc{nullptr};
    torch::nn::LayerNorm input_ln{nullptr};
    torch::nn::ModuleList global_res_blocks{nullptr};

    // Aggregation layers — fuse local + global signals
    torch::nn::Linear fuse_fc{nullptr};
    torch::nn::LayerNorm fuse_ln{nullptr};

    // Policy head
    torch::nn::Linear p_fc1{nullptr}, p_fc2{nullptr};
    torch::nn::LayerNorm p_ln{nullptr};

    // Value head
    torch::nn::Linear v_fc1{nullptr}, v_fc2{nullptr}, v_fc3{nullptr};
    torch::nn::LayerNorm v_ln1{nullptr}, v_ln2{nullptr};

    // Precomputed patch mappings
    std::vector<PatchDesc> patches;

    PatchNetImpl(int big_rows, int big_cols,
                 int patch_rows, int patch_cols,
                 int local_hidden = 128,
                 int local_blocks = 6,
                 int global_hidden = 192,
                 int global_blocks = 4,
                 double dropout = 0.1)
        : big_rows(big_rows), big_cols(big_cols),
          patch_rows(patch_rows), patch_cols(patch_cols),
          n_patches_r(big_rows - patch_rows + 1),
          n_patches_c(big_cols - patch_cols + 1),
          n_patches(n_patches_r * n_patches_c),
          global_action_size((big_rows + 1) * big_cols + big_rows * (big_cols + 1)),
          local_action_size((patch_rows + 1) * patch_cols + patch_rows * (patch_cols + 1))
    {
        // --- Compute feature sizes ---
        const int big_n_h = (big_rows + 1) * big_cols;
        const int big_n_v = big_rows * (big_cols + 1);
        const int big_n_box = big_rows * big_cols;
        global_feat_size = big_n_h + big_n_v + big_n_box + big_n_box * 5 + 2;

        const int loc_n_h = (patch_rows + 1) * patch_cols;
        const int loc_n_v = patch_rows * (patch_cols + 1);
        const int loc_n_box = patch_rows * patch_cols;
        local_feat_size = loc_n_h + loc_n_v + loc_n_box + loc_n_box * 5 + 2;

        // --- Local model (frozen) ---
        local_model = register_module("local_model",
            AlphaZeroBitNet(patch_rows, patch_cols, local_hidden, local_blocks, dropout));

        // --- Global feature processor ---
        // Input: global features only (patch signals are fused later)
        input_fc = register_module("input_fc",
            torch::nn::Linear(global_feat_size, global_hidden));
        input_ln = register_module("input_ln",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({global_hidden})));

        global_res_blocks = register_module("global_res_blocks",
            torch::nn::ModuleList());
        for (int i = 0; i < global_blocks; i++) {
            global_res_blocks->push_back(FCResidualBlock(global_hidden, dropout));
        }

        // --- Fusion layer ---
        // Input: global_hidden + local_policy_aggregate + local_value_signals
        const int fuse_input = global_hidden + global_action_size + n_patches;
        fuse_fc = register_module("fuse_fc",
            torch::nn::Linear(fuse_input, global_hidden));
        fuse_ln = register_module("fuse_ln",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({global_hidden})));

        // --- Policy head ---
        p_fc1 = register_module("p_fc1",
            torch::nn::Linear(global_hidden, global_hidden / 2));
        p_ln = register_module("p_ln",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({global_hidden / 2})));
        p_fc2 = register_module("p_fc2",
            torch::nn::Linear(global_hidden / 2, global_action_size));

        // --- Value head ---
        v_fc1 = register_module("v_fc1",
            torch::nn::Linear(global_hidden, global_hidden / 2));
        v_ln1 = register_module("v_ln1",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({global_hidden / 2})));
        v_fc2 = register_module("v_fc2",
            torch::nn::Linear(global_hidden / 2, global_hidden / 4));
        v_ln2 = register_module("v_ln2",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({global_hidden / 4})));
        v_fc3 = register_module("v_fc3",
            torch::nn::Linear(global_hidden / 4, 1));

        // --- Precompute patch positions & index mappings ---
        precompute_patches();
    }

    /// Load pre-trained local model weights and freeze them.
    void load_and_freeze_local(const std::string& path) {
        torch::load(local_model, path);
        for (auto& param : local_model->parameters()) {
            param.set_requires_grad(false);
        }
        local_model->eval();
        std::cerr << "[PatchNet] Loaded & froze local model: " << path << std::endl;
    }

    /// Freeze local model params (call after loading).
    void freeze_local() {
        for (auto& param : local_model->parameters()) {
            param.set_requires_grad(false);
        }
        local_model->eval();
    }

    // ── Preprocess ───────────────────────────────────────────────────
    // Returns: [global_features | patch_0_features | ... | patch_N_features]
    // Size: global_feat_size + n_patches * local_feat_size

    int combined_input_size() const {
        return global_feat_size + n_patches * local_feat_size;
    }

    static torch::Tensor preprocess(const StateSnapshot& state,
                                    int rows, int cols,
                                    int p_rows, int p_cols,
                                    const std::vector<PatchDesc>& patch_descs) {
        // 1. Global features
        auto global_feat = AlphaZeroBitNetImpl::preprocess(state, rows, cols);

        // 2. Patch features
        std::vector<torch::Tensor> parts;
        parts.reserve(1 + patch_descs.size());
        parts.push_back(global_feat);

        for (const auto& pd : patch_descs) {
            StateSnapshot local_state = extract_patch(state, pd,
                                                       rows, cols,
                                                       p_rows, p_cols);
            auto local_feat = AlphaZeroBitNetImpl::preprocess(
                local_state, p_rows, p_cols);
            parts.push_back(local_feat);
        }

        return torch::cat(parts);
    }

    /// Instance convenience overload.
    torch::Tensor preprocess(const StateSnapshot& state) {
        return PatchNetImpl::preprocess(state, big_rows, big_cols,
                                         patch_rows, patch_cols, patches);
    }

    // ── Forward ──────────────────────────────────────────────────────
    // Input: (batch, combined_input_size)
    // Output: {policy_logits: (batch, global_action_size), value: (batch, 1)}

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        const int B = x.size(0);

        // ── 1. Split global and patch features ──
        auto global_feats = x.narrow(1, 0, global_feat_size);          // (B, G)
        auto patch_feats  = x.narrow(1, global_feat_size,
                                      n_patches * local_feat_size);    // (B, P*L)

        // ── 2. Run local model on all patches (batched, no grad) ──
        torch::Tensor local_logits, local_values;
        {
            torch::NoGradGuard no_grad;  // freeze local model computation
            auto flat = patch_feats.reshape({B * n_patches, local_feat_size});
            auto [logits, values] = local_model->forward(flat);
            local_logits = logits.reshape({B, n_patches, local_action_size});
            local_values = values.reshape({B, n_patches});
        }

        // ── 3. Scatter local policy to global action space ──
        // For each patch, map local action logits to their global positions.
        // Overlapping edges get averaged.
        auto policy_map = torch::zeros({B, global_action_size},
                                        x.options());  // (B, A)
        auto overlap_count = torch::zeros({global_action_size},
                                           torch::kFloat32).to(x.device());

        for (int p = 0; p < n_patches; p++) {
            auto patch_logits = local_logits.select(1, p);  // (B, local_action_size)
            const auto& mapping = patches[p].local_to_global;
            for (int l = 0; l < local_action_size; l++) {
                const int g = mapping[l];
                policy_map.select(1, g) += patch_logits.select(1, l);
                if (p == 0) {
                    // Count overlaps (same for all samples in batch)
                    // We'll accumulate and fix after the loop
                }
            }
        }
        // Build overlap count once
        for (int p = 0; p < n_patches; p++) {
            for (int l = 0; l < local_action_size; l++) {
                const int g = patches[p].local_to_global[l];
                overlap_count[g] += 1.0f;
            }
        }
        // Average overlapping signals
        policy_map = policy_map / overlap_count.unsqueeze(0).clamp_min(1.0f);

        // ── 4. Process global features through residual tower ──
        auto h = torch::relu(input_ln(input_fc(global_feats)));
        for (size_t i = 0; i < global_res_blocks->size(); i++) {
            h = global_res_blocks->ptr<FCResidualBlockImpl>(i)->forward(h);
        }

        // ── 5. Fuse global hidden + local policy + local values ──
        auto fused = torch::cat({h, policy_map, local_values}, /*dim=*/1);
        auto f = torch::relu(fuse_ln(fuse_fc(fused)));

        // ── 6. Policy head ──
        auto p = torch::relu(p_ln(p_fc1(f)));
        auto policy_logits = p_fc2(p);

        // ── 7. Value head ──
        auto v = torch::relu(v_ln1(v_fc1(f)));
        v = torch::relu(v_ln2(v_fc2(v)));
        auto value = torch::tanh(v_fc3(v));

        return {policy_logits, value};
    }

private:
    // ── Patch precomputation ─────────────────────────────────────────

    void precompute_patches() {
        patches.clear();
        patches.reserve(n_patches);

        const int big_n_h_cols = big_cols;
        const int big_n_v_cols = big_cols + 1;
        const int loc_n_h_cols = patch_cols;
        const int loc_n_v_cols = patch_cols + 1;
        const int big_h_total = (big_rows + 1) * big_cols;

        for (int pr = 0; pr < n_patches_r; pr++) {
            for (int pc = 0; pc < n_patches_c; pc++) {
                PatchDesc pd;
                pd.r0 = pr;
                pd.c0 = pc;
                pd.local_to_global.resize(local_action_size);

                // H-edges: local (lr, lc) → global (lr+pr, lc+pc)
                // local range: lr ∈ [0, patch_rows], lc ∈ [0, patch_cols-1]
                const int loc_n_h = (patch_rows + 1) * patch_cols;
                for (int i = 0; i < loc_n_h; i++) {
                    const int lr = i / loc_n_h_cols;
                    const int lc = i % loc_n_h_cols;
                    const int gr = lr + pr;
                    const int gc = lc + pc;
                    const int global_idx = gr * big_n_h_cols + gc;
                    pd.local_to_global[i] = global_idx;
                }

                // V-edges: local (lr, lc) → global (lr+pr, lc+pc)
                // local range: lr ∈ [0, patch_rows-1], lc ∈ [0, patch_cols]
                const int loc_n_v = patch_rows * (patch_cols + 1);
                for (int i = 0; i < loc_n_v; i++) {
                    const int lr = i / loc_n_v_cols;
                    const int lc = i % loc_n_v_cols;
                    const int gr = lr + pr;
                    const int gc = lc + pc;
                    const int global_idx = big_h_total + gr * big_n_v_cols + gc;
                    pd.local_to_global[loc_n_h + i] = global_idx;
                }

                patches.push_back(std::move(pd));
            }
        }

        std::cerr << "[PatchNet] " << n_patches << " patches ("
                  << patch_rows << "x" << patch_cols << " on "
                  << big_rows << "x" << big_cols << ")" << std::endl;
    }

    // ── Extract a KxK patch state from the full NxN board ────────────

    static StateSnapshot extract_patch(const StateSnapshot& full,
                                        const PatchDesc& pd,
                                        int big_rows, int big_cols,
                                        int p_rows, int p_cols) {
        const int r0 = pd.r0, c0 = pd.c0;

        // Local bitset sizes
        const int loc_n_h = (p_rows + 1) * p_cols;
        const int loc_n_v = p_rows * (p_cols + 1);
        const int loc_n_box = p_rows * p_cols;

        FastBitset loc_h(loc_n_h);
        FastBitset loc_v(loc_n_v);
        FastBitset loc_bp1(loc_n_box);
        FastBitset loc_bp2(loc_n_box);

        // Copy h-edges
        for (int lr = 0; lr <= p_rows; lr++) {
            for (int lc = 0; lc < p_cols; lc++) {
                const int gr = lr + r0, gc = lc + c0;
                const int gidx = gr * big_cols + gc;
                if (full.h_edges.test(gidx)) {
                    loc_h.set(lr * p_cols + lc);
                }
            }
        }

        // Copy v-edges
        for (int lr = 0; lr < p_rows; lr++) {
            for (int lc = 0; lc <= p_cols; lc++) {
                const int gr = lr + r0, gc = lc + c0;
                const int gidx = gr * (big_cols + 1) + gc;
                if (full.v_edges.test(gidx)) {
                    loc_v.set(lr * (p_cols + 1) + lc);
                }
            }
        }

        // Copy box ownership
        int loc_score_p1 = 0, loc_score_p2 = 0;
        for (int lr = 0; lr < p_rows; lr++) {
            for (int lc = 0; lc < p_cols; lc++) {
                const int gr = lr + r0, gc = lc + c0;
                const int gidx = gr * big_cols + gc;
                const int lidx = lr * p_cols + lc;
                if (full.boxes_p1.test(gidx)) {
                    loc_bp1.set(lidx);
                    loc_score_p1++;
                }
                if (full.boxes_p2.test(gidx)) {
                    loc_bp2.set(lidx);
                    loc_score_p2++;
                }
            }
        }

        return StateSnapshot{
            std::move(loc_h), std::move(loc_v),
            std::move(loc_bp1), std::move(loc_bp2),
            full.current_player, false,
            loc_score_p1, loc_score_p2,
            Action{-1, -1, -1}
        };
    }
};
TORCH_MODULE(PatchNet);

}  // namespace azb
