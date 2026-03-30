#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace azb {

struct Action {
    int edge_type = 0;  // 0 = horizontal, 1 = vertical
    int r = 0;
    int c = 0;
};

struct StepResult {
    uint64_t boxes_p1 = 0;
    uint64_t boxes_p2 = 0;
    int score_p1 = 0;
    int score_p2 = 0;
    bool done = false;
    int reward = 0;
};

struct StateSnapshot {
    uint64_t h_edges = 0;
    uint64_t v_edges = 0;
    uint64_t boxes_p1 = 0;
    uint64_t boxes_p2 = 0;
    int current_player = 1;
    bool done = false;
    int score_p1 = 0;
    int score_p2 = 0;
    Action last_action{};
};

struct ActionRecord {
    Action action{};
    int player = 1;
    bool box_made = false;
    int reward = 0;
};

/// Dots-and-Boxes environment using bitmask state.
/// Supports rectangular boards: rows x cols boxes.
/// For a square NxN board, use BitBoardEnv(N) or BitBoardEnv(N, N).
class BitBoardEnv {
public:
    /// Square board: N x N boxes.
    explicit BitBoardEnv(int n);
    /// Rectangular board: rows x cols boxes.
    BitBoardEnv(int rows, int cols);

    void reset();
    StepResult step(const Action& action);
    std::vector<Action> get_available_actions() const;
    uint64_t get_legal_actions_mask() const;
    BitBoardEnv clone() const;
    StateSnapshot get_state_snapshot(const Action& action) const;
    std::string render() const;

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int current_player() const { return current_player_; }
    bool done() const { return done_; }
    int score_p1() const { return score_[0]; }
    int score_p2() const { return score_[1]; }
    uint64_t h_edges() const { return h_edges_; }
    uint64_t v_edges() const { return v_edges_; }
    uint64_t boxes_p1() const { return boxes_p1_; }
    uint64_t boxes_p2() const { return boxes_p2_; }
    int n_h_edges() const { return n_h_edges_; }
    int n_v_edges() const { return n_v_edges_; }
    int total_boxes() const { return total_boxes_; }
    int action_size() const { return n_h_edges_ + n_v_edges_; }

    /// Directly set internal state (used by server for Python interop).
    void set_state(uint64_t h_edges, uint64_t v_edges,
                   uint64_t boxes_p1, uint64_t boxes_p2,
                   int current_player, bool done,
                   int score_p1, int score_p2) {
        h_edges_ = h_edges;
        v_edges_ = v_edges;
        boxes_p1_ = boxes_p1;
        boxes_p2_ = boxes_p2;
        current_player_ = current_player;
        done_ = done;
        score_[0] = score_p1;
        score_[1] = score_p2;
    }

public:
    struct Masks {
        std::vector<uint64_t> box_h_masks;
        std::vector<uint64_t> box_v_masks;
        std::vector<std::vector<std::pair<int, int>>> h_edge_adj_boxes;
        std::vector<std::vector<std::pair<int, int>>> v_edge_adj_boxes;
        uint64_t all_h = 0ULL;
        uint64_t all_v = 0ULL;
    };

private:
    void init_common();
    static Masks precompute_masks(int rows, int cols);
    static uint64_t make_cache_key(int rows, int cols) {
        return (static_cast<uint64_t>(rows) << 32) | static_cast<uint64_t>(cols);
    }

    int rows_ = 0;
    int cols_ = 0;
    int n_h_edges_ = 0;
    int n_v_edges_ = 0;
    int total_edges_ = 0;
    int total_boxes_ = 0;
    uint64_t h_edges_ = 0;
    uint64_t v_edges_ = 0;
    uint64_t boxes_p1_ = 0;
    uint64_t boxes_p2_ = 0;
    int current_player_ = 1;
    bool done_ = false;
    int score_[2] = {0, 0};

    std::vector<ActionRecord> action_history_;
    std::vector<StateSnapshot> state_history_;

    static std::unordered_map<uint64_t, Masks> masks_cache_;
    const Masks* masks_ = nullptr;
};

}  // namespace azb
