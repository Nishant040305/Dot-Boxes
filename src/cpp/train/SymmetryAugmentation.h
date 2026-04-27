#pragma once
/// Symmetry augmentation for Dots-and-Boxes training data.
///
/// Generates transformed copies of (state, policy, mask, value) tuples
/// by applying board symmetries.  For rectangular boards, the valid
/// symmetries are horizontal flip and vertical flip (2 transforms, 3×
/// data).  For square boards, the full D₄ dihedral group applies:
/// identity, 2 reflections, 3 rotations, 2 diagonal flips (7 transforms,
/// 8× data).
///
/// All transforms are zero-cost (no NN calls) — pure index remapping.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>
#include "AlphaZeroBitNet.h"
#include "BitBoardEnv.h"
#include "ReplayBuffer.h"

namespace azb {

/// Board geometry constants, computed once per worker.
struct BoardGeometry {
    int rows, cols;
    int n_h_edges;   // (rows+1) * cols
    int n_v_edges;   // rows * (cols+1)
    int n_boxes;     // rows * cols
    int action_size; // n_h_edges + n_v_edges
    bool is_square;

    BoardGeometry(int r, int c)
        : rows(r), cols(c),
          n_h_edges((r + 1) * c),
          n_v_edges(r * (c + 1)),
          n_boxes(r * c),
          action_size(n_h_edges + n_v_edges),
          is_square(r == c) {}
};

// ─── Transform descriptor ────────────────────────────────────────────
// A transform maps every edge and box index to its new position.
// Two kinds:
//   • Reflection: h→h, v→v  (edge types preserved)
//   • Rotation:   h→v, v→h  (edge types swapped)

enum class TransformKind { kReflection, kRotation };

/// Pre-computed index mapping tables for one symmetry transform.
struct SymmetryTransform {
    TransformKind kind;

    // For kReflection: h_map[i] = new h-edge index, v_map[i] = new v-edge index
    // For kRotation:   h_map[i] = new v-edge index, v_map[i] = new h-edge index
    std::vector<size_t> h_map;
    std::vector<size_t> v_map;
    std::vector<size_t> box_map;
};

// ─── Build all transforms for a given board ──────────────────────────

inline std::vector<SymmetryTransform> build_transforms(const BoardGeometry& geo) {
    std::vector<SymmetryTransform> transforms;
    const int R = geo.rows;
    const int C = geo.cols;
    const int n_h = geo.n_h_edges;
    const int n_v = geo.n_v_edges;
    const int n_b = geo.n_boxes;

    // Helper to build h-edge remap table (type-preserving)
    auto make_h_table = [&](auto fn) {
        std::vector<size_t> tbl(static_cast<size_t>(n_h));
        for (int i = 0; i < n_h; i++) {
            int r = i / C, c = i % C;
            tbl[i] = fn(r, c);
        }
        return tbl;
    };
    // Helper to build v-edge remap table (type-preserving)
    auto make_v_table = [&](auto fn) {
        std::vector<size_t> tbl(static_cast<size_t>(n_v));
        for (int i = 0; i < n_v; i++) {
            int r = i / (C + 1), c = i % (C + 1);
            tbl[i] = fn(r, c);
        }
        return tbl;
    };
    // Helper to build box remap table
    auto make_box_table = [&](auto fn) {
        std::vector<size_t> tbl(static_cast<size_t>(n_b));
        for (int i = 0; i < n_b; i++) {
            int r = i / C, c = i % C;
            tbl[i] = fn(r, c);
        }
        return tbl;
    };

    // ── 1. Horizontal flip (left↔right) ──────────────────────────────
    // Works for any rectangular board.
    {
        SymmetryTransform t;
        t.kind = TransformKind::kReflection;
        // h-edge (r,c) → (r, C-1-c)
        t.h_map = make_h_table([&](int r, int c) -> size_t {
            return static_cast<size_t>(r * C + (C - 1 - c));
        });
        // v-edge (r,c) → (r, C-c)
        t.v_map = make_v_table([&](int r, int c) -> size_t {
            return static_cast<size_t>(r * (C + 1) + (C - c));
        });
        // box (r,c) → (r, C-1-c)
        t.box_map = make_box_table([&](int r, int c) -> size_t {
            return static_cast<size_t>(r * C + (C - 1 - c));
        });
        transforms.push_back(std::move(t));
    }

    // ── 2. Vertical flip (top↔bottom) ────────────────────────────────
    {
        SymmetryTransform t;
        t.kind = TransformKind::kReflection;
        // h-edge (r,c) → (R-r, c)
        t.h_map = make_h_table([&](int r, int c) -> size_t {
            return static_cast<size_t>((R - r) * C + c);
        });
        // v-edge (r,c) → (R-1-r, c)
        t.v_map = make_v_table([&](int r, int c) -> size_t {
            return static_cast<size_t>((R - 1 - r) * (C + 1) + c);
        });
        // box (r,c) → (R-1-r, c)
        t.box_map = make_box_table([&](int r, int c) -> size_t {
            return static_cast<size_t>((R - 1 - r) * C + c);
        });
        transforms.push_back(std::move(t));
    }

    // ── Square-only transforms (D₄ group extras) ─────────────────────
    if (!geo.is_square) return transforms;

    const int N = R;  // R == C for square boards

    // ── 3. 180° rotation (h→h, v→v, types preserved) ────────────────
    {
        SymmetryTransform t;
        t.kind = TransformKind::kReflection;
        // h-edge (r,c) → (N-r, N-1-c)
        t.h_map = make_h_table([&](int r, int c) -> size_t {
            return static_cast<size_t>((N - r) * N + (N - 1 - c));
        });
        // v-edge (r,c) → (N-1-r, N-c)
        t.v_map = make_v_table([&](int r, int c) -> size_t {
            return static_cast<size_t>((N - 1 - r) * (N + 1) + (N - c));
        });
        // box (r,c) → (N-1-r, N-1-c)
        t.box_map = make_box_table([&](int r, int c) -> size_t {
            return static_cast<size_t>((N - 1 - r) * N + (N - 1 - c));
        });
        transforms.push_back(std::move(t));
    }

    // Helper for rotation-type transforms (h↔v swap).
    // h_map[i] stores the v-edge index that h-edge i maps to.
    // v_map[i] stores the h-edge index that v-edge i maps to.
    auto make_h_to_v_table = [&](auto fn) {
        std::vector<size_t> tbl(static_cast<size_t>(n_h));
        for (int i = 0; i < n_h; i++) {
            int r = i / N, c = i % N;
            tbl[i] = fn(r, c);
        }
        return tbl;
    };
    auto make_v_to_h_table = [&](auto fn) {
        std::vector<size_t> tbl(static_cast<size_t>(n_v));
        for (int i = 0; i < n_v; i++) {
            int r = i / (N + 1), c = i % (N + 1);
            tbl[i] = fn(r, c);
        }
        return tbl;
    };

    // ── 4. 90° CW rotation ──────────────────────────────────────────
    //   dot (r,c) → (c, N-r)
    //   h-edge (r,c) → v-edge (c, N-r)
    //   v-edge (r,c) → h-edge (c, N-1-r)
    //   box    (r,c) → box    (c, N-1-r)
    {
        SymmetryTransform t;
        t.kind = TransformKind::kRotation;
        t.h_map = make_h_to_v_table([&](int r, int c) -> size_t {
            return static_cast<size_t>(c * (N + 1) + (N - r));
        });
        t.v_map = make_v_to_h_table([&](int r, int c) -> size_t {
            return static_cast<size_t>(c * N + (N - 1 - r));
        });
        t.box_map = make_box_table([&](int r, int c) -> size_t {
            return static_cast<size_t>(c * N + (N - 1 - r));
        });
        transforms.push_back(std::move(t));
    }

    // ── 5. 270° CW rotation (= 90° CCW) ────────────────────────────
    //   dot (r,c) → (N-c, r)
    //   h-edge (r,c) → v-edge (N-1-c, r)
    //   v-edge (r,c) → h-edge (N-c, r)
    //   box    (r,c) → box    (N-1-c, r)
    {
        SymmetryTransform t;
        t.kind = TransformKind::kRotation;
        t.h_map = make_h_to_v_table([&](int r, int c) -> size_t {
            return static_cast<size_t>((N - 1 - c) * (N + 1) + r);
        });
        t.v_map = make_v_to_h_table([&](int r, int c) -> size_t {
            return static_cast<size_t>((N - c) * N + r);
        });
        t.box_map = make_box_table([&](int r, int c) -> size_t {
            return static_cast<size_t>((N - 1 - c) * N + r);
        });
        transforms.push_back(std::move(t));
    }

    // ── 6. Transpose (p1 diagonal) ────────────────────────────────
    //   dot (r,c) → (c, r)
    //   h-edge (r,c) → v-edge (c, r)
    //   v-edge (r,c) → h-edge (c, r)
    //   box    (r,c) → box    (c, r)
    {
        SymmetryTransform t;
        t.kind = TransformKind::kRotation;
        t.h_map = make_h_to_v_table([&](int r, int c) -> size_t {
            return static_cast<size_t>(c * (N + 1) + r);
        });
        t.v_map = make_v_to_h_table([&](int r, int c) -> size_t {
            return static_cast<size_t>(c * N + r);
        });
        t.box_map = make_box_table([&](int r, int c) -> size_t {
            return static_cast<size_t>(c * N + r);
        });
        transforms.push_back(std::move(t));
    }

    // ── 7. Anti-diagonal flip ───────────────────────────────────────
    //   dot (r,c) → (N-c, N-r)
    //   h-edge (r,c) → v-edge (N-1-c, N-r)
    //   v-edge (r,c) → h-edge (N-c, N-1-r)
    //   box    (r,c) → box    (N-1-c, N-1-r)
    {
        SymmetryTransform t;
        t.kind = TransformKind::kRotation;
        t.h_map = make_h_to_v_table([&](int r, int c) -> size_t {
            return static_cast<size_t>((N - 1 - c) * (N + 1) + (N - r));
        });
        t.v_map = make_v_to_h_table([&](int r, int c) -> size_t {
            return static_cast<size_t>((N - c) * N + (N - 1 - r));
        });
        t.box_map = make_box_table([&](int r, int c) -> size_t {
            return static_cast<size_t>((N - 1 - c) * N + (N - 1 - r));
        });
        transforms.push_back(std::move(t));
    }

    return transforms;
}

// ─── Apply one transform to produce a training sample ────────────────

/// Given a game position and a symmetry transform, produce a new
/// TrainSample with transformed state, policy, and legal mask.
/// The value target `z` is unchanged (symmetry-invariant).
inline TrainSample apply_symmetry(
        const SymmetryTransform& xform,
        const BoardGeometry& geo,
        const StateSnapshot& snapshot,
        const std::unordered_map<uint32_t, int>& visits,
        int total_visits,
        float z,
        float policy_target_temp = 1.0f) {

    const int n_h = geo.n_h_edges;
    const int n_v = geo.n_v_edges;
    const int n_b = geo.n_boxes;

    StateSnapshot out;
    out.h_edges    = azb::FastBitset(n_h);
    out.v_edges    = azb::FastBitset(n_v);
    out.boxes_p1   = azb::FastBitset(n_b);
    out.boxes_p2   = azb::FastBitset(n_b);
    out.current_player = snapshot.current_player;
    out.done           = snapshot.done;
    out.score_p1       = snapshot.score_p1;
    out.score_p2       = snapshot.score_p2;
    out.last_action    = snapshot.last_action;

    if (xform.kind == TransformKind::kReflection) {
        // h→h, v→v
        snapshot.h_edges.for_each_set_bit([&](size_t i) {
            out.h_edges.set(xform.h_map[i]);
        });
        snapshot.v_edges.for_each_set_bit([&](size_t i) {
            out.v_edges.set(xform.v_map[i]);
        });
    } else {
        // h→v, v→h (rotation / diagonal)
        snapshot.h_edges.for_each_set_bit([&](size_t i) {
            out.v_edges.set(xform.h_map[i]);
        });
        snapshot.v_edges.for_each_set_bit([&](size_t i) {
            out.h_edges.set(xform.v_map[i]);
        });
    }

    snapshot.boxes_p1.for_each_set_bit([&](size_t i) {
        out.boxes_p1.set(xform.box_map[i]);
    });
    snapshot.boxes_p2.for_each_set_bit([&](size_t i) {
        out.boxes_p2.set(xform.box_map[i]);
    });

    // Remap policy visit counts
    auto policy = torch::zeros({geo.action_size}, torch::kFloat32);
    auto mask   = torch::zeros({geo.action_size}, torch::kFloat32);
    auto p_acc  = policy.accessor<float, 1>();
    auto m_acc  = mask.accessor<float, 1>();

    const float inv_temp = 1.0f / std::max(policy_target_temp, 0.01f);
    const bool sharpen = std::abs(inv_temp - 1.0f) > 1e-6f;

    for (const auto& [idx, v] : visits) {
        int new_idx;
        if (xform.kind == TransformKind::kReflection) {
            if (static_cast<int>(idx) < n_h) {
                new_idx = static_cast<int>(xform.h_map[idx]);
            } else {
                new_idx = n_h + static_cast<int>(xform.v_map[idx - n_h]);
            }
        } else {
            // Rotation: h-actions → v-actions, v-actions → h-actions
            if (static_cast<int>(idx) < n_h) {
                new_idx = n_h + static_cast<int>(xform.h_map[idx]);
            } else {
                new_idx = static_cast<int>(xform.v_map[idx - n_h]);
            }
        }
        if (!sharpen) {
            p_acc[new_idx] += static_cast<float>(v) / total_visits;
        } else {
            // Store raw visit count; we'll sharpen below in log-space
            p_acc[new_idx] += static_cast<float>(v);
        }
        m_acc[new_idx]  = 1.0f;
    }

    // Re-normalize (with optional log-space sharpening to avoid overflow)
    if (sharpen) {
        // Log-space sharpening: v^inv_temp = exp(inv_temp * log(v))
        // Subtract max before exp for numerical stability (log-sum-exp trick).
        float max_log = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < geo.action_size; i++) {
            if (p_acc[i] > 0.0f) {
                float lv = inv_temp * std::log(p_acc[i]);
                p_acc[i] = lv;
                if (lv > max_log) max_log = lv;
            }
        }
        float psum = 0.0f;
        for (int i = 0; i < geo.action_size; i++) {
            if (m_acc[i] > 0.0f) {
                p_acc[i] = std::exp(p_acc[i] - max_log);
                psum += p_acc[i];
            }
        }
        if (psum > 0.0f) {
            for (int i = 0; i < geo.action_size; i++) p_acc[i] /= psum;
        }
    }

    auto state_tensor = AlphaZeroBitNetImpl::preprocess(out, geo.rows, geo.cols);
    return {std::move(state_tensor), std::move(policy), std::move(mask), z};
}

/// Generate all symmetry-augmented samples for one game position.
/// Appends to `out_samples`.
inline void augment_position(
        const std::vector<SymmetryTransform>& transforms,
        const BoardGeometry& geo,
        const StateSnapshot& snapshot,
        const std::unordered_map<uint32_t, int>& visits,
        int total_visits,
        float z,
        std::vector<TrainSample>& out_samples,
        float policy_target_temp = 1.0f) {

    if (total_visits <= 0) return;
    for (const auto& xform : transforms) {
        out_samples.push_back(
            apply_symmetry(xform, geo, snapshot, visits, total_visits, z,
                           policy_target_temp));
    }
}

}  // namespace azb
