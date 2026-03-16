#include "BitBoardEnv.h"

#include <sstream>
#include <stdexcept>

namespace azb {

std::unordered_map<int, BitBoardEnv::Masks> BitBoardEnv::masks_cache_;

BitBoardEnv::BitBoardEnv(int n) : n_(n) {
    if (n_ <= 0) {
        throw std::invalid_argument("N must be positive");
    }
    n_h_edges_ = (n_ + 1) * n_;
    n_v_edges_ = n_ * (n_ + 1);
    total_edges_ = n_h_edges_ + n_v_edges_;
    total_boxes_ = n_ * n_;

    if (masks_cache_.find(n_) == masks_cache_.end()) {
        masks_cache_[n_] = precompute_masks(n_);
    }
    masks_ = &masks_cache_.at(n_);
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
        ? (1ULL << static_cast<uint64_t>(r * n_ + c))
        : (1ULL << static_cast<uint64_t>(r * (n_ + 1) + c));

    const std::vector<std::pair<int, int>>& adj_boxes =
        (edge_type == 0) ? masks_->h_edge_adj_boxes[r * n_ + c]
                         : masks_->v_edge_adj_boxes[r * (n_ + 1) + c];

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
        const uint64_t h_mask = masks_->box_h_masks[box_r * n_ + box_c];
        const uint64_t v_mask = masks_->box_v_masks[box_r * n_ + box_c];
        const uint64_t box_bit = 1ULL << static_cast<uint64_t>(box_r * n_ + box_c);

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
        const int r = idx / n_;
        const int c = idx % n_;
        actions.push_back({0, r, c});
        tmp ^= bit;
    }

    tmp = free_v;
    while (tmp) {
        const uint64_t bit = tmp & (~tmp + 1);
        const int idx = static_cast<int>(__builtin_ctzll(tmp));
        const int r = idx / (n_ + 1);
        const int c = idx % (n_ + 1);
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
    BitBoardEnv env(n_);
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
    const int dots = n_ + 1;

    for (int r = 0; r < dots; r++) {
        for (int c = 0; c < n_; c++) {
            out << ".";
            const int h_idx = r * n_ + c;
            const bool has = (h_edges_ >> h_idx) & 1ULL;
            out << (has ? "---" : "   ");
        }
        out << ".\n";

        if (r == n_) break;

        for (int c = 0; c < n_ + 1; c++) {
            const int v_idx = r * (n_ + 1) + c;
            const bool has = (v_edges_ >> v_idx) & 1ULL;
            out << (has ? "|" : " ");
            if (c < n_) {
                const int box_idx = r * n_ + c;
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

BitBoardEnv::Masks BitBoardEnv::precompute_masks(int n) {
    Masks masks;
    masks.box_h_masks.resize(n * n, 0ULL);
    masks.box_v_masks.resize(n * n, 0ULL);
    masks.h_edge_adj_boxes.resize((n + 1) * n);
    masks.v_edge_adj_boxes.resize(n * (n + 1));

    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
            const int h_top = r * n + c;
            const int h_bot = (r + 1) * n + c;
            const int v_left = r * (n + 1) + c;
            const int v_right = r * (n + 1) + (c + 1);
            masks.box_h_masks[r * n + c] = (1ULL << h_top) | (1ULL << h_bot);
            masks.box_v_masks[r * n + c] = (1ULL << v_left) | (1ULL << v_right);
        }
    }

    for (int r = 0; r < n + 1; r++) {
        for (int c = 0; c < n; c++) {
            std::vector<std::pair<int, int>> adj;
            if (r > 0) adj.push_back({r - 1, c});
            if (r < n) adj.push_back({r, c});
            masks.h_edge_adj_boxes[r * n + c] = std::move(adj);
        }
    }

    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n + 1; c++) {
            std::vector<std::pair<int, int>> adj;
            if (c > 0) adj.push_back({r, c - 1});
            if (c < n) adj.push_back({r, c});
            masks.v_edge_adj_boxes[r * (n + 1) + c] = std::move(adj);
        }
    }

    masks.all_h = (n == 0) ? 0ULL : ((1ULL << ((n + 1) * n)) - 1ULL);
    masks.all_v = (n == 0) ? 0ULL : ((1ULL << (n * (n + 1))) - 1ULL);
    return masks;
}

}  // namespace azb
