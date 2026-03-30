#include "BitBoardEnv.h"

#include <sstream>
#include <stdexcept>

namespace azb {

std::unordered_map<uint64_t, BitBoardEnv::Masks> BitBoardEnv::masks_cache_;

BitBoardEnv::BitBoardEnv(int n) : rows_(n), cols_(n) { init_common(); }

BitBoardEnv::BitBoardEnv(int rows, int cols) : rows_(rows), cols_(cols) { init_common(); }

void BitBoardEnv::init_common() {
    if (rows_ <= 0 || cols_ <= 0) {
        throw std::invalid_argument("rows and cols must be positive");
    }
    n_h_edges_ = (rows_ + 1) * cols_;
    n_v_edges_ = rows_ * (cols_ + 1);
    total_edges_ = n_h_edges_ + n_v_edges_;
    total_boxes_ = rows_ * cols_;

    const uint64_t key = make_cache_key(rows_, cols_);
    if (masks_cache_.find(key) == masks_cache_.end()) {
        masks_cache_[key] = precompute_masks(rows_, cols_);
    }
    masks_ = &masks_cache_.at(key);
    reset();
}

void BitBoardEnv::reset() {
    h_edges_ = 0ULL;
    v_edges_ = 0ULL;
    boxes_p1_ = 0ULL;
    boxes_p2_ = 0ULL;
    current_player_ = 1;
    done_ = false;
    score_[0] = 0;
    score_[1] = 0;
    action_history_.clear();
    state_history_.clear();
    state_history_.push_back(get_state_snapshot(Action{-1, -1, -1}));
}

StepResult BitBoardEnv::step(const Action& action) {
    if (done_) {
        return {boxes_p1_, boxes_p2_, score_[0], score_[1], true, 0};
    }

    const int player_before = current_player_;
    const int edge_type = action.edge_type;
    const int r = action.r;
    const int c = action.c;
    const uint64_t bit = (edge_type == 0)
        ? (1ULL << static_cast<uint64_t>(r * cols_ + c))
        : (1ULL << static_cast<uint64_t>(r * (cols_ + 1) + c));

    const std::vector<std::pair<int, int>>& adj_boxes =
        (edge_type == 0) ? masks_->h_edge_adj_boxes[r * cols_ + c]
                         : masks_->v_edge_adj_boxes[r * (cols_ + 1) + c];

    if (edge_type == 0) {
        h_edges_ |= bit;
    } else {
        v_edges_ |= bit;
    }

    bool box_made = false;
    int reward = 0;

    for (const auto& box : adj_boxes) {
        const int box_r = box.first;
        const int box_c = box.second;
        const uint64_t h_mask = masks_->box_h_masks[box_r * cols_ + box_c];
        const uint64_t v_mask = masks_->box_v_masks[box_r * cols_ + box_c];
        const uint64_t box_bit = 1ULL << static_cast<uint64_t>(box_r * cols_ + box_c);

        if ((h_edges_ & h_mask) == h_mask && (v_edges_ & v_mask) == v_mask) {
            if ((boxes_p1_ & box_bit) == 0 && (boxes_p2_ & box_bit) == 0) {
                if (current_player_ == 1) {
                    boxes_p1_ |= box_bit;
                } else {
                    boxes_p2_ |= box_bit;
                }
                score_[current_player_ - 1] += 1;
                reward += 1;
                box_made = true;
            }
        }
    }

    if (score_[0] + score_[1] == total_boxes_) {
        done_ = true;
    }
    if (!box_made) {
        current_player_ = 3 - current_player_;
    }

    action_history_.push_back({action, player_before, box_made, reward});
    state_history_.push_back(get_state_snapshot(action));

    return {boxes_p1_, boxes_p2_, score_[0], score_[1], done_, reward};
}

std::vector<Action> BitBoardEnv::get_available_actions() const {
    std::vector<Action> actions;
    const uint64_t free_h = masks_->all_h & ~h_edges_;
    const uint64_t free_v = masks_->all_v & ~v_edges_;

    uint64_t tmp = free_h;
    while (tmp) {
        const uint64_t bit = tmp & (~tmp + 1);
        const int idx = static_cast<int>(__builtin_ctzll(tmp));
        const int r = idx / cols_;
        const int c = idx % cols_;
        actions.push_back({0, r, c});
        tmp ^= bit;
    }

    tmp = free_v;
    while (tmp) {
        const uint64_t bit = tmp & (~tmp + 1);
        const int idx = static_cast<int>(__builtin_ctzll(tmp));
        const int r = idx / (cols_ + 1);
        const int c = idx % (cols_ + 1);
        actions.push_back({1, r, c});
        tmp ^= bit;
    }

    return actions;
}

uint64_t BitBoardEnv::get_legal_actions_mask() const {
    const uint64_t free_h = masks_->all_h & ~h_edges_;
    const uint64_t free_v = masks_->all_v & ~v_edges_;
    return free_h | (free_v << n_h_edges_);
}

BitBoardEnv BitBoardEnv::clone() const {
    BitBoardEnv env(rows_, cols_);
    env.h_edges_ = h_edges_;
    env.v_edges_ = v_edges_;
    env.boxes_p1_ = boxes_p1_;
    env.boxes_p2_ = boxes_p2_;
    env.current_player_ = current_player_;
    env.done_ = done_;
    env.score_[0] = score_[0];
    env.score_[1] = score_[1];
    env.action_history_.clear();
    env.state_history_.clear();
    return env;
}

StateSnapshot BitBoardEnv::get_state_snapshot(const Action& action) const {
    StateSnapshot snap;
    snap.h_edges = h_edges_;
    snap.v_edges = v_edges_;
    snap.boxes_p1 = boxes_p1_;
    snap.boxes_p2 = boxes_p2_;
    snap.current_player = current_player_;
    snap.done = done_;
    snap.score_p1 = score_[0];
    snap.score_p2 = score_[1];
    snap.last_action = action;
    return snap;
}

std::string BitBoardEnv::render() const {
    std::ostringstream out;

    for (int r = 0; r < rows_ + 1; r++) {
        for (int c = 0; c < cols_; c++) {
            out << ".";
            const int h_idx = r * cols_ + c;
            const bool has = (h_edges_ >> h_idx) & 1ULL;
            out << (has ? "---" : "   ");
        }
        out << ".\n";

        if (r == rows_) break;

        for (int c = 0; c < cols_ + 1; c++) {
            const int v_idx = r * (cols_ + 1) + c;
            const bool has = (v_edges_ >> v_idx) & 1ULL;
            out << (has ? "|" : " ");
            if (c < cols_) {
                const int box_idx = r * cols_ + c;
                char owner = ' ';
                if ((boxes_p1_ >> box_idx) & 1ULL) owner = '1';
                if ((boxes_p2_ >> box_idx) & 1ULL) owner = '2';
                out << " " << owner << " ";
            }
        }
        out << "\n";
    }

    return out.str();
}

BitBoardEnv::Masks BitBoardEnv::precompute_masks(int rows, int cols) {
    Masks masks;
    masks.box_h_masks.resize(rows * cols, 0ULL);
    masks.box_v_masks.resize(rows * cols, 0ULL);
    masks.h_edge_adj_boxes.resize((rows + 1) * cols);
    masks.v_edge_adj_boxes.resize(rows * (cols + 1));

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            const int h_top = r * cols + c;
            const int h_bot = (r + 1) * cols + c;
            const int v_left = r * (cols + 1) + c;
            const int v_right = r * (cols + 1) + (c + 1);
            masks.box_h_masks[r * cols + c] = (1ULL << h_top) | (1ULL << h_bot);
            masks.box_v_masks[r * cols + c] = (1ULL << v_left) | (1ULL << v_right);
        }
    }

    for (int r = 0; r < rows + 1; r++) {
        for (int c = 0; c < cols; c++) {
            std::vector<std::pair<int, int>> adj;
            if (r > 0) adj.push_back({r - 1, c});
            if (r < rows) adj.push_back({r, c});
            masks.h_edge_adj_boxes[r * cols + c] = std::move(adj);
        }
    }

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols + 1; c++) {
            std::vector<std::pair<int, int>> adj;
            if (c > 0) adj.push_back({r, c - 1});
            if (c < cols) adj.push_back({r, c});
            masks.v_edge_adj_boxes[r * (cols + 1) + c] = std::move(adj);
        }
    }

    const int nh = (rows + 1) * cols;
    const int nv = rows * (cols + 1);
    masks.all_h = (nh == 0) ? 0ULL : ((1ULL << nh) - 1ULL);
    masks.all_v = (nv == 0) ? 0ULL : ((1ULL << nv) - 1ULL);
    return masks;
}

}  // namespace azb
