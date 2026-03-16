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

class BitBoardEnv {
public:
    explicit BitBoardEnv(int n);

    void reset();
    StepResult step(const Action& action);
    std::vector<Action> get_available_actions() const;
    uint64_t get_legal_actions_mask() const;
    BitBoardEnv clone() const;
    StateSnapshot get_state_snapshot(const Action& action) const;
    std::string render() const;

    int N() const { return n_; }
    int current_player() const { return current_player_; }
    bool done() const { return done_; }
    int score_p1() const { return score_[0]; }
    int score_p2() const { return score_[1]; }
    uint64_t h_edges() const { return h_edges_; }
    uint64_t v_edges() const { return v_edges_; }
    uint64_t boxes_p1() const { return boxes_p1_; }
    uint64_t boxes_p2() const { return boxes_p2_; }

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
    static Masks precompute_masks(int n);

    int n_ = 0;
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

    static std::unordered_map<int, Masks> masks_cache_;
    const Masks* masks_ = nullptr;
};

}  // namespace azb
