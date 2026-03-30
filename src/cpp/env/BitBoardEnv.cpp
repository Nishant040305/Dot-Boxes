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

    h_edges_.resize(n_h_edges_);
    v_edges_.resize(n_v_edges_);
    boxes_p1_.resize(total_boxes_);
    boxes_p2_.resize(total_boxes_);

    const uint64_t key = make_cache_key(rows_, cols_);
    if (masks_cache_.find(key) == masks_cache_.end()) {
        masks_cache_[key] = precompute_masks(rows_, cols_, n_h_edges_, n_v_edges_, total_boxes_);
    }
    masks_ = &masks_cache_.at(key);
    reset();
}

void BitBoardEnv::reset() {
    h_edges_.clear();
    v_edges_.clear();
    boxes_p1_.clear();
    boxes_p2_.clear();
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

    const std::vector<std::pair<int, int>>& adj_boxes =
        (edge_type == 0) ? masks_->h_edge_adj_boxes[r * cols_ + c]
                         : masks_->v_edge_adj_boxes[r * (cols_ + 1) + c];

    if (edge_type == 0) {
        h_edges_.set(r * cols_ + c);
    } else {
        v_edges_.set(r * (cols_ + 1) + c);
    }

    bool box_made = false;
    int reward = 0;

    for (const auto& box : adj_boxes) {
        const int box_r = box.first;
        const int box_c = box.second;
        const auto& h_mask = masks_->box_h_masks[box_r * cols_ + box_c];
        const auto& v_mask = masks_->box_v_masks[box_r * cols_ + box_c];
        const int box_bit = box_r * cols_ + box_c;

        if (h_edges_.contains_all(h_mask) && v_edges_.contains_all(v_mask)) {
            if (!boxes_p1_.test(box_bit) && !boxes_p2_.test(box_bit)) {
                if (current_player_ == 1) {
                    boxes_p1_.set(box_bit);
                } else {
                    boxes_p2_.set(box_bit);
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
    azb::FastBitset free_h = masks_->all_h.bit_and(h_edges_.bit_not());
    azb::FastBitset free_v = masks_->all_v.bit_and(v_edges_.bit_not());

    free_h.for_each_set_bit([&](size_t idx) {
        actions.push_back({0, static_cast<int>(idx) / cols_, static_cast<int>(idx) % cols_});
    });

    free_v.for_each_set_bit([&](size_t idx) {
        actions.push_back({1, static_cast<int>(idx) / (cols_ + 1), static_cast<int>(idx) % (cols_ + 1)});
    });

    return actions;
}

azb::FastBitset BitBoardEnv::get_legal_actions_mask() const {
    azb::FastBitset free_h = masks_->all_h.bit_and(h_edges_.bit_not());
    azb::FastBitset free_v = masks_->all_v.bit_and(v_edges_.bit_not());
    azb::FastBitset combined(action_size());
    
    free_h.for_each_set_bit([&](size_t idx) { combined.set(idx); });
    free_v.for_each_set_bit([&](size_t idx) { combined.set(n_h_edges_ + idx); });
    
    return combined;
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
            const bool has = h_edges_.test(h_idx);
            out << (has ? "---" : "   ");
        }
        out << ".\n";

        if (r == rows_) break;

        for (int c = 0; c < cols_ + 1; c++) {
            const int v_idx = r * (cols_ + 1) + c;
            const bool has = v_edges_.test(v_idx);
            out << (has ? "|" : " ");
            if (c < cols_) {
                const int box_idx = r * cols_ + c;
                char owner = ' ';
                if (boxes_p1_.test(box_idx)) owner = '1';
                if (boxes_p2_.test(box_idx)) owner = '2';
                out << " " << owner << " ";
            }
        }
        out << "\n";
    }

    return out.str();
}

BitBoardEnv::Masks BitBoardEnv::precompute_masks(int rows, int cols, int n_h, int n_v, int t_boxes) {
    Masks masks;
    masks.box_h_masks.assign(t_boxes, azb::FastBitset(n_h));
    masks.box_v_masks.assign(t_boxes, azb::FastBitset(n_v));
    masks.h_edge_adj_boxes.resize(n_h);
    masks.v_edge_adj_boxes.resize(n_v);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            const int h_top = r * cols + c;
            const int h_bot = (r + 1) * cols + c;
            const int v_left = r * (cols + 1) + c;
            const int v_right = r * (cols + 1) + (c + 1);
            masks.box_h_masks[r * cols + c].set(h_top);
            masks.box_h_masks[r * cols + c].set(h_bot);
            masks.box_v_masks[r * cols + c].set(v_left);
            masks.box_v_masks[r * cols + c].set(v_right);
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

    masks.all_h.resize(n_h);
    masks.all_h.set_all();
    masks.all_v.resize(n_v);
    masks.all_v.set_all();
    return masks;
}

}  // namespace azb
