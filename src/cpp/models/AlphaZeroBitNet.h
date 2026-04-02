#pragma once
/// AlphaZeroBitNet — LibTorch neural network for bitboard-based AlphaZero.
///
/// Architecture: FC residual blocks → policy head (logits) + value head (tanh).
/// Supports rectangular boards (rows x cols).
///
/// Input features (flat vector):
///   - Horizontal edge bits:    (rows+1)*cols
///   - Vertical edge bits:      rows*(cols+1)
///   - Box ownership:           rows*cols  (-1/0/+1, current-player relative)
///   - Edge-count one-hot:      rows*cols * 5  (0,1,2,3,4 edges per box)
///   - boxes_filled / total:    1  (game-phase scalar, 0→1)
///   - score_diff / total:      1  (current player score advantage, normalised)
///
/// Total input_size = (rows+1)*cols + rows*(cols+1) + rows*cols + rows*cols*5 + 2

#include <torch/torch.h>
#include "BitBoardEnv.h"

namespace azb {

// ─── FC Residual Block ───────────────────────────────────────────────

struct FCResidualBlockImpl : torch::nn::Module {
    FCResidualBlockImpl(int hidden_size, double dropout_rate = 0.1)
        : fc1(register_module("fc1", torch::nn::Linear(hidden_size, hidden_size))),
          ln1(register_module("ln1", torch::nn::LayerNorm(
              torch::nn::LayerNormOptions({hidden_size})))),
          fc2(register_module("fc2", torch::nn::Linear(hidden_size, hidden_size))),
          ln2(register_module("ln2", torch::nn::LayerNorm(
              torch::nn::LayerNormOptions({hidden_size})))),
          dropout(register_module("dropout", torch::nn::Dropout(dropout_rate))) {}

    torch::Tensor forward(torch::Tensor x) {
        auto residual = x;
        auto out = torch::relu(ln1(fc1(x)));
        out = dropout(out);
        out = ln2(fc2(out));
        out = out + residual;
        out = torch::relu(out);
        return out;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr};
    torch::nn::Dropout dropout{nullptr};
};
TORCH_MODULE(FCResidualBlock);

// ─── AlphaZero BitNet ────────────────────────────────────────────────

struct AlphaZeroBitNetImpl : torch::nn::Module {
    int rows, cols;
    int n_h_edges, n_v_edges, action_size, input_size;

    torch::nn::Linear input_fc{nullptr};
    torch::nn::LayerNorm input_ln{nullptr};
    torch::nn::ModuleList res_blocks{nullptr};

    // Policy head
    torch::nn::Linear p_fc1{nullptr}, p_fc2{nullptr};
    torch::nn::LayerNorm p_ln{nullptr};

    // Value head — deeper: hidden → h/2 → h/4 → 1
    torch::nn::Linear v_fc1{nullptr}, v_fc2{nullptr}, v_fc3{nullptr};
    torch::nn::LayerNorm v_ln{nullptr}, v_ln2{nullptr};

    AlphaZeroBitNetImpl(int rows, int cols,
                        int hidden_size = 256,
                        int num_res_blocks = 6,
                        double dropout = 0.1)
        : rows(rows), cols(cols),
          n_h_edges((rows + 1) * cols),
          n_v_edges(rows * (cols + 1)),
          action_size(n_h_edges + n_v_edges),
          input_size(n_h_edges + n_v_edges + rows * cols   // raw edges + box ownership
                     + rows * cols * 5                     // edge-count one-hot (5 ch/box)
                     + 2) {                                // game-progress scalars
        // input_size breakdown:
        //  (rows+1)*cols  h-edge bits
        //  rows*(cols+1)  v-edge bits
        //  rows*cols      box ownership (+1/−1 relative to current player)
        //  rows*cols * 5  one-hot edge count per box (0,1,2,3,4)
        //  1              boxes_filled / total_boxes  (game-phase, 0→1)
        //  1              score_diff   / total_boxes  (cur-player advantage, −1→+1)

        // Input projection
        input_fc = register_module("input_fc",
            torch::nn::Linear(input_size, hidden_size));
        input_ln = register_module("input_ln",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size})));

        // Residual tower
        res_blocks = register_module("res_blocks", torch::nn::ModuleList());
        for (int i = 0; i < num_res_blocks; i++) {
            res_blocks->push_back(FCResidualBlock(hidden_size, dropout));
        }

        // Policy head
        p_fc1 = register_module("p_fc1",
            torch::nn::Linear(hidden_size, hidden_size / 2));
        p_ln = register_module("p_ln",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size / 2})));
        p_fc2 = register_module("p_fc2",
            torch::nn::Linear(hidden_size / 2, action_size));

        // Value head — two hidden layers.
        // The value head needs to aggregate global game state (total boxes,
        // chain parity) which requires more capacity than a single linear layer.
        v_fc1 = register_module("v_fc1",
            torch::nn::Linear(hidden_size, hidden_size / 2));
        v_ln = register_module("v_ln",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size / 2})));
        v_fc2 = register_module("v_fc2",
            torch::nn::Linear(hidden_size / 2, hidden_size / 4));
        v_ln2 = register_module("v_ln2",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size / 4})));
        v_fc3 = register_module("v_fc3",
            torch::nn::Linear(hidden_size / 4, 1));
    }

    /// Forward pass.
    /// x: (batch, input_size) → returns {policy_logits: (batch, action_size), value: (batch, 1)}
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        // Input projection
        x = torch::relu(input_ln(input_fc(x)));

        // Residual tower
        for (size_t i = 0; i < res_blocks->size(); i++) {
            x = res_blocks->ptr<FCResidualBlockImpl>(i)->forward(x);
        }

        // Policy head
        auto p = torch::relu(p_ln(p_fc1(x)));
        auto policy_logits = p_fc2(p);

        // Value head: hidden → h/2 → h/4 → tanh
        auto v = torch::relu(v_ln(v_fc1(x)));
        v = torch::relu(v_ln2(v_fc2(v)));
        auto value = torch::tanh(v_fc3(v));

        return {policy_logits, value};
    }

    /// Compute edge count (0-4) for a single box at (r,c).
    /// This is the #1 Dots-and-Boxes heuristic feature:
    ///   count=3  -> free box (one move away from a point)
    ///   count=2  -> chain/half-open box (dangerous to play)
    static int edge_count(const StateSnapshot& state, int r, int c, int cols) {
        const int n_h = (r > 0 ? 1 : 0) +     // top edge present?
                        (1);                    // always has bottom
        // We compute by just counting bits. Inline for speed.
        int count = 0;
        // horizontal: top = r*cols+c, bottom = (r+1)*cols+c
        if (state.h_edges.test(static_cast<size_t>(r * cols + c)))       count++;
        if (state.h_edges.test(static_cast<size_t>((r + 1) * cols + c))) count++;
        // vertical: left = r*(cols+1)+c, right = r*(cols+1)+(c+1)
        if (state.v_edges.test(static_cast<size_t>(r * (cols + 1) + c)))       count++;
        if (state.v_edges.test(static_cast<size_t>(r * (cols + 1) + (c + 1)))) count++;
        (void)n_h;
        return count;
    }

    /// Convert a single StateSnapshot to a feature tensor.
    /// Layout: [h_edges | v_edges | box_owner | edge_count_onehot(5 ch) | progress(2)]
    static torch::Tensor preprocess(const StateSnapshot& state, int rows, int cols) {
        const int n_h   = (rows + 1) * cols;
        const int n_v   = rows * (cols + 1);
        const int n_box = rows * cols;
        // 5 one-hot channels per box + 2 game-progress scalars
        const int in_sz = n_h + n_v + n_box + n_box * 5 + 2;

        auto features = torch::zeros({in_sz}, torch::kFloat32);
        float* ptr = features.data_ptr<float>();

        // Edge bits
        state.h_edges.for_each_set_bit([&](size_t i) { ptr[i] = 1.0f; });
        state.v_edges.for_each_set_bit([&](size_t i) { ptr[n_h + i] = 1.0f; });

        // Box ownership (current-player-relative)
        const int box_off  = n_h + n_v;
        state.boxes_p1.for_each_set_bit([&](size_t i) {
            ptr[box_off + i] = (state.current_player == 1) ? 1.0f : -1.0f;
        });
        state.boxes_p2.for_each_set_bit([&](size_t i) {
            ptr[box_off + i] = (state.current_player == 2) ? 1.0f : -1.0f;
        });

        // Edge-count one-hot (5 channels per box)
        // channel k at position: (n_h + n_v + n_box) + box_idx*5 + k
        const int cnt_off = box_off + n_box;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                const int box_idx = r * cols + c;
                const int k = edge_count(state, r, c, cols);  // 0..4
                ptr[cnt_off + box_idx * 5 + k] = 1.0f;
            }
        }

        // Game-progress scalars (2 features)
        // These encode global game phase directly, relieving the value head
        // from having to sum ownership bits across all boxes.
        const float total_f = static_cast<float>(n_box);
        const int filled = static_cast<int>(state.boxes_p1.popcount() +
                                            state.boxes_p2.popcount());
        const int my_score  = (state.current_player == 1) ? state.score_p1 : state.score_p2;
        const int opp_score = (state.current_player == 1) ? state.score_p2 : state.score_p1;
        const int prog_off = cnt_off + n_box * 5;
        ptr[prog_off]     = static_cast<float>(filled) / total_f;          // 0→1 game phase
        ptr[prog_off + 1] = static_cast<float>(my_score - opp_score) / total_f; // advantage

        return features;
    }

    /// Instance convenience overload.
    torch::Tensor preprocess(const StateSnapshot& state) {
        return AlphaZeroBitNetImpl::preprocess(state, rows, cols);
    }

    /// Batch preprocess from raw bitmask arrays.
    torch::Tensor preprocess_batch(
            const std::vector<azb::FastBitset>& h_edges_list,
            const std::vector<azb::FastBitset>& v_edges_list,
            const std::vector<azb::FastBitset>& boxes_p1_list,
            const std::vector<azb::FastBitset>& boxes_p2_list,
            const std::vector<int>& players_list) {
        const int batch  = static_cast<int>(h_edges_list.size());
        const int n_box  = rows * cols;
        const int cnt_off_base = n_h_edges + n_v_edges + n_box;
        auto features = torch::zeros({batch, input_size}, torch::kFloat32);
        auto acc = features.accessor<float, 2>();

        for (int b = 0; b < batch; b++) {
            const int player = players_list[b];

            h_edges_list[b].for_each_set_bit([&](size_t i) { acc[b][i] = 1.0f; });
            v_edges_list[b].for_each_set_bit([&](size_t i) { acc[b][n_h_edges + i] = 1.0f; });

            const int box_off_b = n_h_edges + n_v_edges;
            boxes_p1_list[b].for_each_set_bit([&](size_t i) {
                acc[b][box_off_b + i] = (player == 1) ? 1.0f : -1.0f;
            });
            boxes_p2_list[b].for_each_set_bit([&](size_t i) {
                acc[b][box_off_b + i] = (player == 2) ? 1.0f : -1.0f;
            });

            // Edge-count one-hot per box
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    const int box_idx = r * cols + c;
                    int count = 0;
                    if (h_edges_list[b].test(static_cast<size_t>(r * cols + c)))           count++;
                    if (h_edges_list[b].test(static_cast<size_t>((r+1) * cols + c)))       count++;
                    if (v_edges_list[b].test(static_cast<size_t>(r * (cols+1) + c)))       count++;
                    if (v_edges_list[b].test(static_cast<size_t>(r * (cols+1) + (c+1))))   count++;
                    acc[b][cnt_off_base + box_idx * 5 + count] = 1.0f;
                }
            }

            // Game-progress scalars
            const float total_f = static_cast<float>(n_box);
            const int filled_b = static_cast<int>(boxes_p1_list[b].popcount() +
                                                   boxes_p2_list[b].popcount());
            // We track scores via popcount of box bitsets
            const int p1_score = static_cast<int>(boxes_p1_list[b].popcount());
            const int p2_score = static_cast<int>(boxes_p2_list[b].popcount());
            const int my_score_b  = (player == 1) ? p1_score : p2_score;
            const int opp_score_b = (player == 1) ? p2_score : p1_score;
            const int prog_off_b = cnt_off_base + n_box * 5;
            acc[b][prog_off_b]     = static_cast<float>(filled_b) / total_f;
            acc[b][prog_off_b + 1] = static_cast<float>(my_score_b - opp_score_b) / total_f;
        }
        return features;
    }
};
TORCH_MODULE(AlphaZeroBitNet);

}  // namespace azb
