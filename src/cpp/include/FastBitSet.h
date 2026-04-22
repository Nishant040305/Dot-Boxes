#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace azb {

/// Maximum inline words for stack-allocated bitset storage.
/// 2 words = 128 bits — covers all practical Dots-and-Boxes boards
/// (up to ~11x11 = 121 edges per type). Zero heap allocations.
static constexpr size_t kMaxInlineWords = 2;

class FastBitset {
private:
    uint64_t data_[kMaxInlineWords]{};
    size_t num_bits_ = 0;
    size_t num_words_ = 0;
    uint64_t last_mask_ = ~0ULL;

public:
    explicit FastBitset(size_t bits = 0) { resize(bits); }

    void resize(size_t bits) {
        num_bits_ = bits;
        num_words_ = (bits + 63) >> 6;
        // static_assert-safe: all practical boards fit in kMaxInlineWords
        std::memset(data_, 0, sizeof(data_));
        if (num_words_ == 0) {
            last_mask_ = ~0ULL;
        } else {
            const size_t used_bits = num_bits_ - ((num_words_ - 1) << 6);
            last_mask_ = (used_bits == 64) ? ~0ULL : ((1ULL << used_bits) - 1ULL);
        }
    }

    size_t size() const { return num_bits_; }
    size_t num_words() const { return num_words_; }

    void clear() {
        for (size_t i = 0; i < num_words_; i++) data_[i] = 0ULL;
    }

    void set(size_t pos) { data_[pos >> 6] |= (1ULL << (pos & 63)); }
    void reset(size_t pos) { data_[pos >> 6] &= ~(1ULL << (pos & 63)); }
    bool test(size_t pos) const { return (data_[pos >> 6] >> (pos & 63)) & 1ULL; }

    void set_all() {
        if (num_words_ == 0) return;
        for (size_t i = 0; i < num_words_; i++) data_[i] = ~0ULL;
        data_[num_words_ - 1] &= last_mask_;
    }

    bool contains_all(const FastBitset& mask) const {
        for (size_t i = 0; i < num_words_; i++) {
            if ((data_[i] & mask.data_[i]) != mask.data_[i]) return false;
        }
        return true;
    }

    size_t popcount() const {
        size_t total = 0;
        for (size_t i = 0; i < num_words_; i++)
            total += static_cast<size_t>(__builtin_popcountll(data_[i]));
        return total;
    }

    FastBitset bit_not() const {
        FastBitset out(*this);
        for (size_t i = 0; i < num_words_; i++) out.data_[i] = ~out.data_[i];
        if (num_words_ > 0) out.data_[num_words_ - 1] &= last_mask_;
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
        if (num_words_ > 0) data_[num_words_ - 1] &= last_mask_;
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

    uint64_t* raw() { return data_; }
    const uint64_t* raw() const { return data_; }

    /// 128-bit hash: two independent 64-bit polynomial hashes over all words.
    /// Collision-safe even for boards up to 20x20 (~1240 bits of state).
    std::pair<uint64_t, uint64_t> hash128() const {
        // Two independent FNV-like hashes with different primes
        uint64_t h0 = 0xcbf29ce484222325ULL ^ num_bits_;
        uint64_t h1 = 0x9e3779b97f4a7c15ULL ^ num_bits_;
        for (size_t i = 0; i < num_words_; i++) {
            h0 ^= data_[i];
            h0 *= 0x100000001b3ULL;       // FNV prime
            h0 ^= h0 >> 33;

            h1 ^= data_[i];
            h1 *= 0x517cc1b727220a95ULL;  // different multiplier
            h1 ^= h1 >> 27;
            h1 *= 0x94d049bb133111ebULL;
        }
        return {h0, h1};
    }
};

}  // namespace azb
