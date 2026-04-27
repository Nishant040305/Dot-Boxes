#pragma once
/// AlphaZeroCNNNet — Convolutional Neural Network for bitboard-based AlphaZero.
///
/// Architecture: 2D-CNN residual blocks → policy head (logits) + value head (tanh).
/// Supports rectangular boards (rows x cols).
///
/// Input features (spatial tensor, C × H × W):
///   H = rows + 1, W = cols + 1 (dots-grid dimensions)
///   Channels:
///     0:  Horizontal edges      (1 if edge placed, 0 otherwise)
///     1:  Vertical edges        (1 if edge placed, 0 otherwise)
///     2:  Box ownership — me    (+1 for current player's boxes)
///     3:  Box ownership — opp   (+1 for opponent's boxes)
///     4-8: Edge-count one-hot   (channels for 0,1,2,3,4 edges per box)
///     9:  Game progress         (boxes_filled / total, broadcast)
///    10:  Score difference       (my_score - opp_score / total, broadcast)
///
///   Total channels = 11

#include <torch/torch.h>
#include "BitBoardEnv.h"

namespace azb {

// Number of input feature channels for CNN preprocessing
constexpr int kCNNInputChannels = 11;

// ─── Conv2d Residual Block ───────────────────────────────────────────

struct ConvResidualBlockImpl : torch::nn::Module {
    ConvResidualBlockImpl(int channels, double dropout_rate = 0.1)
        : conv1(register_module("conv1",
              torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1)))),
          bn1(register_module("bn1", torch::nn::BatchNorm2d(channels))),
          conv2(register_module("conv2",
              torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1)))),
          bn2(register_module("bn2", torch::nn::BatchNorm2d(channels))),
          dropout(register_module("dropout", torch::nn::Dropout(dropout_rate))) {}

    torch::Tensor forward(torch::Tensor x) {
        auto residual = x;
        auto out = torch::relu(bn1(conv1(x)));
        out = dropout(out);
        out = bn2(conv2(out));
        out = out + residual;
        out = torch::relu(out);
        return out;
    }

    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    torch::nn::Dropout dropout{nullptr};
};
TORCH_MODULE(ConvResidualBlock);

// ─── AlphaZero CNN Net ──────────────────────────────────────────────

struct AlphaZeroCNNNetImpl : torch::nn::Module {
    int rows, cols;
    int grid_h, grid_w;  // rows+1, cols+1 (dot-grid dimensions)
    int n_h_edges, n_v_edges, action_size;
    int cnn_channels;

    // Input convolution
    torch::nn::Conv2d input_conv{nullptr};
    torch::nn::BatchNorm2d input_bn{nullptr};

    // Residual tower
    torch::nn::ModuleList res_blocks{nullptr};

    // Policy head (conv → flatten → FC → FC)
    torch::nn::Conv2d p_conv{nullptr};
    torch::nn::BatchNorm2d p_bn{nullptr};
    torch::nn::Linear p_fc1{nullptr}, p_fc2{nullptr};
    torch::nn::BatchNorm1d p_bn_fc{nullptr};

    // Value head (conv → flatten → FC → FC → FC → tanh)
    torch::nn::Conv2d v_conv{nullptr};
    torch::nn::BatchNorm2d v_bn{nullptr};
    torch::nn::Linear v_fc1{nullptr}, v_fc2{nullptr}, v_fc3{nullptr};
    torch::nn::LayerNorm v_ln1{nullptr}, v_ln2{nullptr};

    AlphaZeroCNNNetImpl(int rows, int cols,
                        int cnn_channels = 128,
                        int num_res_blocks = 6,
                        double dropout = 0.1)
        : rows(rows), cols(cols),
          grid_h(rows + 1), grid_w(cols + 1),
          n_h_edges((rows + 1) * cols),
          n_v_edges(rows * (cols + 1)),
          action_size(n_h_edges + n_v_edges),
          cnn_channels(cnn_channels) {

        // Input projection: 11 channels → cnn_channels
        input_conv = register_module("input_conv",
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(kCNNInputChannels, cnn_channels, 3).padding(1)));
        input_bn = register_module("input_bn",
            torch::nn::BatchNorm2d(cnn_channels));

        // Residual tower
        res_blocks = register_module("res_blocks", torch::nn::ModuleList());
        for (int i = 0; i < num_res_blocks; i++) {
            res_blocks->push_back(ConvResidualBlock(cnn_channels, dropout));
        }

        // Policy head: conv 1×1 → BN → flatten → FC hidden → BN → FC output
        //   Use 32 conv channels (not 2) so flatten gives 32*H*W = 1152
        //   features, then a hidden FC layer for non-linear action mixing.
        const int p_channels = 32;
        p_conv = register_module("p_conv",
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(cnn_channels, p_channels, 1)));
        p_bn = register_module("p_bn",
            torch::nn::BatchNorm2d(p_channels));
        const int p_flat = p_channels * grid_h * grid_w;
        const int p_hidden = cnn_channels;  // hidden layer width
        p_fc1 = register_module("p_fc1",
            torch::nn::Linear(p_flat, p_hidden));
        p_bn_fc = register_module("p_bn_fc",
            torch::nn::BatchNorm1d(p_hidden));
        p_fc2 = register_module("p_fc2",
            torch::nn::Linear(p_hidden, action_size));

        // Value head: conv 1×1 → BN → flatten → FC → LN → FC → LN → FC → tanh
        //   Use 4 conv channels for richer spatial summarisation.
        const int v_channels = 4;
        v_conv = register_module("v_conv",
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(cnn_channels, v_channels, 1)));
        v_bn = register_module("v_bn",
            torch::nn::BatchNorm2d(v_channels));

        const int v_flat = v_channels * grid_h * grid_w;
        const int v_hidden = cnn_channels;  // match channel count
        v_fc1 = register_module("v_fc1",
            torch::nn::Linear(v_flat, v_hidden));
        v_ln1 = register_module("v_ln1",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({v_hidden})));
        v_fc2 = register_module("v_fc2",
            torch::nn::Linear(v_hidden, v_hidden / 4));
        v_ln2 = register_module("v_ln2",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({v_hidden / 4})));
        v_fc3 = register_module("v_fc3",
            torch::nn::Linear(v_hidden / 4, 1));
    }

    /// Forward pass.
    /// x: (batch, 11, grid_h, grid_w) → returns {policy_logits, value}
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        // Input projection
        x = torch::relu(input_bn(input_conv(x)));

        // Residual tower
        for (size_t i = 0; i < res_blocks->size(); i++) {
            x = res_blocks->ptr<ConvResidualBlockImpl>(i)->forward(x);
        }

        // Policy head
        auto p = torch::relu(p_bn(p_conv(x)));
        p = p.flatten(1);  // (batch, p_channels * H * W)
        p = torch::relu(p_bn_fc(p_fc1(p)));
        auto policy_logits = p_fc2(p);

        // Value head
        auto v = torch::relu(v_bn(v_conv(x)));
        v = v.flatten(1);
        v = torch::relu(v_ln1(v_fc1(v)));
        v = torch::relu(v_ln2(v_fc2(v)));
        auto value = torch::tanh(v_fc3(v));

        return {policy_logits, value};
    }

    /// Convert a StateSnapshot to a spatial feature tensor.
    /// Returns shape: (11, grid_h, grid_w)
    static torch::Tensor preprocess(const StateSnapshot& state, int rows, int cols) {
        const int H = rows + 1;
        const int W = cols + 1;

        // (C, H, W) tensor — 11 channels
        auto features = torch::zeros({kCNNInputChannels, H, W}, torch::kFloat32);
        auto acc = features.accessor<float, 3>();

        // Channel 0: Horizontal edges
        // h_edge(r, c) placed at spatial position (r, c) — r ∈ [0, rows], c ∈ [0, cols-1]
        for (int r = 0; r <= rows; r++) {
            for (int c = 0; c < cols; c++) {
                const int idx = r * cols + c;
                if (state.h_edges.test(static_cast<size_t>(idx))) {
                    acc[0][r][c] = 1.0f;
                }
            }
        }

        // Channel 1: Vertical edges
        // v_edge(r, c) placed at spatial position (r, c) — r ∈ [0, rows-1], c ∈ [0, cols]
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c <= cols; c++) {
                const int idx = r * (cols + 1) + c;
                if (state.v_edges.test(static_cast<size_t>(idx))) {
                    acc[1][r][c] = 1.0f;
                }
            }
        }

        // Channels 2-3: Box ownership (current player / opponent)
        // Box (r, c) at spatial position (r, c) — r ∈ [0, rows-1], c ∈ [0, cols-1]
        const bool is_p1 = (state.current_player == 1);
        state.boxes_p1.for_each_set_bit([&](size_t i) {
            const int r = static_cast<int>(i) / cols;
            const int c = static_cast<int>(i) % cols;
            acc[is_p1 ? 2 : 3][r][c] = 1.0f;
        });
        state.boxes_p2.for_each_set_bit([&](size_t i) {
            const int r = static_cast<int>(i) / cols;
            const int c = static_cast<int>(i) % cols;
            acc[is_p1 ? 3 : 2][r][c] = 1.0f;
        });

        // Channels 4-8: Edge count one-hot (0,1,2,3,4) per box
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                int count = 0;
                // top edge
                if (state.h_edges.test(static_cast<size_t>(r * cols + c))) count++;
                // bottom edge
                if (state.h_edges.test(static_cast<size_t>((r + 1) * cols + c))) count++;
                // left edge
                if (state.v_edges.test(static_cast<size_t>(r * (cols + 1) + c))) count++;
                // right edge
                if (state.v_edges.test(static_cast<size_t>(r * (cols + 1) + (c + 1)))) count++;
                acc[4 + count][r][c] = 1.0f;
            }
        }

        // Channel 9: Game progress (broadcast over spatial dims)
        const int n_box = rows * cols;
        const float total_f = static_cast<float>(n_box);
        const int filled = static_cast<int>(state.boxes_p1.popcount() +
                                            state.boxes_p2.popcount());
        const float progress = static_cast<float>(filled) / total_f;
        for (int r = 0; r < H; r++) {
            for (int c = 0; c < W; c++) {
                acc[9][r][c] = progress;
            }
        }

        // Channel 10: Score difference (broadcast)
        const int my_score  = is_p1 ? state.score_p1 : state.score_p2;
        const int opp_score = is_p1 ? state.score_p2 : state.score_p1;
        const float score_diff = static_cast<float>(my_score - opp_score) / total_f;
        for (int r = 0; r < H; r++) {
            for (int c = 0; c < W; c++) {
                acc[10][r][c] = score_diff;
            }
        }

        return features;
    }

    /// Instance convenience overload.
    torch::Tensor preprocess(const StateSnapshot& state) {
        return AlphaZeroCNNNetImpl::preprocess(state, rows, cols);
    }
};
TORCH_MODULE(AlphaZeroCNNNet);

}  // namespace azb
