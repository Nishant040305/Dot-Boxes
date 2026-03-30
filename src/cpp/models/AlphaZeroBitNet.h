#pragma once
/// AlphaZeroBitNet — LibTorch neural network for bitboard-based AlphaZero.
///
/// Architecture: FC residual blocks → policy head (logits) + value head (tanh).
/// Supports rectangular boards (rows x cols).
///
/// Input features (flat vector):
///   - Horizontal edge bits:  (rows+1)*cols
///   - Vertical edge bits:    rows*(cols+1)
///   - Box ownership:         rows*cols  (-1, 0, +1 from current player's perspective)
///   - Current player flag:   1
///
/// Total input_size = (rows+1)*cols + rows*(cols+1) + rows*cols + 1

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

    // Value head
    torch::nn::Linear v_fc1{nullptr}, v_fc2{nullptr};
    torch::nn::LayerNorm v_ln{nullptr};

    AlphaZeroBitNetImpl(int rows, int cols,
                        int hidden_size = 256,
                        int num_res_blocks = 6,
                        double dropout = 0.1)
        : rows(rows), cols(cols),
          n_h_edges((rows + 1) * cols),
          n_v_edges(rows * (cols + 1)),
          action_size(n_h_edges + n_v_edges),
          input_size(n_h_edges + n_v_edges + rows * cols + 1) {

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

        // Value head
        v_fc1 = register_module("v_fc1",
            torch::nn::Linear(hidden_size, hidden_size / 4));
        v_ln = register_module("v_ln",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size / 4})));
        v_fc2 = register_module("v_fc2",
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

        // Value head
        auto v = torch::relu(v_ln(v_fc1(x)));
        auto value = torch::tanh(v_fc2(v));

        return {policy_logits, value};
    }

    /// Convert a single StateSnapshot to a feature tensor.
    /// Static overload — requires no model weights, safe to call without a trained net.
    static torch::Tensor preprocess(const StateSnapshot& state, int rows, int cols) {
        const int n_h   = (rows + 1) * cols;
        const int n_v   = rows * (cols + 1);
        const int n_box = rows * cols;
        const int in_sz = n_h + n_v + n_box + 1;

        auto features = torch::zeros({in_sz}, torch::kFloat32);
        float* ptr = features.data_ptr<float>();

        state.h_edges.for_each_set_bit([&](size_t i) { ptr[i] = 1.0f; });
        state.v_edges.for_each_set_bit([&](size_t i) { ptr[n_h + i] = 1.0f; });

        const int box_offset = n_h + n_v;
        state.boxes_p1.for_each_set_bit([&](size_t i) {
            ptr[box_offset + i] = (state.current_player == 1) ? 1.0f : -1.0f;
        });
        state.boxes_p2.for_each_set_bit([&](size_t i) {
            ptr[box_offset + i] = (state.current_player == 2) ? 1.0f : -1.0f;
        });

        ptr[in_sz - 1] = (state.current_player == 1) ? 1.0f : -1.0f;
        return features;
    }

    /// Instance convenience overload — delegates to the static version.
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
        const int batch = static_cast<int>(h_edges_list.size());
        auto features = torch::zeros({batch, input_size}, torch::kFloat32);
        auto acc = features.accessor<float, 2>();

        for (int b = 0; b < batch; b++) {
            const int player = players_list[b];
            
            h_edges_list[b].for_each_set_bit([&](size_t i) { acc[b][i] = 1.0f; });
            v_edges_list[b].for_each_set_bit([&](size_t i) { acc[b][n_h_edges + i] = 1.0f; });
            
            const int box_off = n_h_edges + n_v_edges;
            boxes_p1_list[b].for_each_set_bit([&](size_t i) {
                acc[b][box_off + i] = (player == 1) ? 1.0f : -1.0f;
            });
            boxes_p2_list[b].for_each_set_bit([&](size_t i) {
                acc[b][box_off + i] = (player == 2) ? 1.0f : -1.0f;
            });
            
            acc[b][input_size - 1] = (player == 1) ? 1.0f : -1.0f;
        }
        return features;
    }
};
TORCH_MODULE(AlphaZeroBitNet);

}  // namespace azb
