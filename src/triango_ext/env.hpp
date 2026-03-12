#pragma once

#include <vector>
#include <array>
#include <cstdint>
#include <string>

// --- BitBoard Structure ---
// Represents the 96-tile grid using two 64-bit integers.
// lo stores bits 0-63. hi stores bits 64-127.
struct BitBoard {
    uint64_t lo = 0;
    uint64_t hi = 0;

    BitBoard() = default;
    BitBoard(uint64_t l, uint64_t h) : lo(l), hi(h) {}

    bool operator==(const BitBoard& other) const {
        return lo == other.lo && hi == other.hi;
    }
    bool operator!=(const BitBoard& other) const {
        return !(*this == other);
    }
    BitBoard operator&(const BitBoard& other) const {
        return BitBoard(lo & other.lo, hi & other.hi);
    }
    BitBoard operator|(const BitBoard& other) const {
        return BitBoard(lo | other.lo, hi | other.hi);
    }
    BitBoard operator~() const {
        return BitBoard(~lo, ~hi);
    }
    BitBoard& operator&=(const BitBoard& other) {
        lo &= other.lo;
        hi &= other.hi;
        return *this;
    }
    BitBoard& operator|=(const BitBoard& other) {
        lo |= other.lo;
        hi |= other.hi;
        return *this;
    }
    bool is_zero() const {
        return lo == 0 && hi == 0;
    }
    int count_ones() const;
    void set_bit(int idx);
    bool get_bit(int idx) const;
};

// --- Core Environment Initialization ---
// Must be called once before instantiating GameStates
void initialize_env();

// --- Globals initialized by initialize_env() ---
extern std::vector<BitBoard> ALL_MASKS;
extern std::vector<std::vector<BitBoard>> STANDARD_PIECES;

// --- GameState ---
struct GameState {
    BitBoard board;
    int score = 0;
    std::vector<int> available;
    int pieces_left = 3;
    bool terminal = false;

    GameState();
    GameState(std::vector<int> pieces, BitBoard board_state, int current_score);

    void check_terminal();
    GameState* apply_move(int slot, int index);
    void refill_tray();

    // Helper functions to pass the BitBoard back and forth with Python
    std::string board_to_string() const;
    static BitBoard string_to_board(const std::string& s);
};
