#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace azb {

class FastBitset {
private:
    std::vector<uint64_t> data_;
    size_t num_bits_ = 0;
    size_t num_words_ = 0;
    uint64_t last_mask_ = ~0ULL;

public:
    explicit FastBitset(size_t bits = 0) { resize(bits); }

    void resize(size_t bits) {
        num_bits_ = bits;
        num_words_ = (bits + 63) >> 6;
        data_.assign(num_words_, 0ULL);
        if (num_words_ == 0) {
            last_mask_ = ~0ULL;
        } else {
            const size_t used_bits = num_bits_ - ((num_words_ - 1) << 6);
            last_mask_ = (used_bits == 64) ? ~0ULL : ((1ULL << used_bits) - 1ULL);
        }
    }

    size_t size() const { return num_bits_; }
    size_t num_words() const { return num_words_; }

    void clear() { std::fill(data_.begin(), data_.end(), 0ULL); }

    void set(size_t pos) { data_[pos >> 6] |= (1ULL << (pos & 63)); }
    void reset(size_t pos) { data_[pos >> 6] &= ~(1ULL << (pos & 63)); }
    bool test(size_t pos) const { return (data_[pos >> 6] >> (pos & 63)) & 1ULL; }

    void set_all() {
        if (num_words_ == 0) return;
        std::fill(data_.begin(), data_.end(), ~0ULL);
        data_.back() &= last_mask_;
    }

    bool contains_all(const FastBitset& mask) const {
        for (size_t i = 0; i < num_words_; i++) {
            if ((data_[i] & mask.data_[i]) != mask.data_[i]) return false;
        }
        return true;
    }

    size_t popcount() const {
        size_t total = 0;
        for (uint64_t v : data_) total += static_cast<size_t>(__builtin_popcountll(v));
        return total;
    }

    FastBitset bit_not() const {
        FastBitset out(*this);
        for (size_t i = 0; i < num_words_; i++) out.data_[i] = ~out.data_[i];
        if (num_words_ > 0) out.data_.back() &= last_mask_;
        return out;
    }

    FastBitset bit_and(const FastBitset& other) const {
        FastBitset out(*this);
        for (size_t i = 0; i < num_words_; i++) out.data_[i] &= other.data_[i];
        return out;
    }

    FastBitset bit_or(const FastBitset& other) const {
        FastBitset out(*this);
        for (size_t i = 0; i < num_words_; i++) out.data_[i] |= other.data_[i];
        return out;
    }

    void and_with(const FastBitset& other) {
        for (size_t i = 0; i < num_words_; i++) data_[i] &= other.data_[i];
    }

    void or_with(const FastBitset& other) {
        for (size_t i = 0; i < num_words_; i++) data_[i] |= other.data_[i];
    }

    void and_not(const FastBitset& other) {
        for (size_t i = 0; i < num_words_; i++) data_[i] &= ~other.data_[i];
        if (num_words_ > 0) data_.back() &= last_mask_;
    }

    template <typename F>
    void for_each_set_bit(F&& f) const {
        for (size_t w = 0; w < num_words_; w++) {
            uint64_t word = data_[w];
            while (word) {
                const int bit = __builtin_ctzll(word);
                f(static_cast<size_t>(w * 64 + bit));
                word &= (word - 1);
            }
        }
    }

    uint64_t* raw() { return data_.data(); }
    const uint64_t* raw() const { return data_.data(); }
};

}  // namespace azb
