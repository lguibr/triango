#include "env.hpp"
#include <iostream>
#include <random>
#include <map>
#include <tuple>
#include <queue>
#include <algorithm>
#include <bit>

// Compatibility macro for popcount (C++20 std::popcount or fallback)
#ifdef __cpp_lib_bitops
#include <bit>
#define POPCOUNT64(x) std::popcount((uint64_t)(x))
#elif defined(_MSC_VER)
#include <intrin.h>
#define POPCOUNT64(x) __popcnt64((uint64_t)(x))
#else
#define POPCOUNT64(x) __builtin_popcountll((uint64_t)(x))
#endif

int BitBoard::count_ones() const {
    return POPCOUNT64(lo) + POPCOUNT64(hi);
}

void BitBoard::set_bit(int idx) {
    if (idx < 64) lo |= (1ULL << idx);
    else hi |= (1ULL << (idx - 64));
}

bool BitBoard::get_bit(int idx) const {
    if (idx < 64) return (lo & (1ULL << idx)) != 0;
    return (hi & (1ULL << (idx - 64))) != 0;
}

// Global lists initialized once
std::vector<BitBoard> ALL_MASKS;
std::vector<std::vector<BitBoard>> STANDARD_PIECES;

static const int TOTAL_TRIANGLES = 96;
static const std::array<int, 8> ROW_LENGTHS = {9, 11, 13, 15, 15, 13, 11, 9};

static int flat_index(int r, int c) {
    int idx = 0;
    for (int i = 0; i < r; ++i) {
        idx += ROW_LENGTHS[i];
    }
    return idx + c;
}

static std::pair<int, int> get_row_col(int idx) {
    int rem = idx;
    for (int r = 0; r < 8; ++r) {
        if (rem < ROW_LENGTHS[r]) return {r, rem};
        rem -= ROW_LENGTHS[r];
    }
    return {-1, -1};
}

static bool is_up(int r, int c) {
    if (r < 4) return c % 2 == 0;
    return c % 2 == 1;
}

static bool is_up_flat(int idx) {
    auto [r, c] = get_row_col(idx);
    return is_up(r, c);
}

static std::pair<int, int> vertical_neighbor(int r, int c) {
    if (is_up(r, c)) {
        if (r == 7) return {-1, -1};
        if (r < 3) return {r + 1, c + 1};
        else if (r == 3) return {r + 1, c};
        else return {r + 1, c - 1};
    } else {
        if (r == 0) return {-1, -1};
        if (r < 4) return {r - 1, c - 1};
        else if (r == 4) return {r - 1, c};
        else return {r - 1, c + 1};
    }
}

// ----------------------------------------------------
// Coordinates Map
struct Coord {
    int x, y, z;
    bool operator<(const Coord& o) const {
        if (x != o.x) return x < o.x;
        if (y != o.y) return y < o.y;
        return z < o.z;
    }
    bool operator==(const Coord& o) const {
        return x == o.x && y == o.y && z == o.z;
    }
};

static std::map<int, Coord> INDEX_TO_COORD;
static std::map<Coord, int> COORD_TO_INDEX;

static void build_coords() {
    std::vector<bool> visited(TOTAL_TRIANGLES, false);
    std::queue<int> q;
    q.push(0);
    INDEX_TO_COORD[0] = {0, 0, 1};
    visited[0] = true;

    auto assign = [&](int n_idx, int nx, int ny, int nz) {
        if (!visited[n_idx]) {
            INDEX_TO_COORD[n_idx] = {nx, ny, nz};
            visited[n_idx] = true;
            q.push(n_idx);
        }
    };

    while (!q.empty()) {
        int curr = q.front();
        q.pop();
        Coord coord = INDEX_TO_COORD[curr];
        auto [r, c] = get_row_col(curr);

        if (is_up(r, c)) {
            if (c + 1 < ROW_LENGTHS[r]) assign(flat_index(r, c + 1), coord.x, coord.y, coord.z - 1);
            if (c - 1 >= 0) assign(flat_index(r, c - 1), coord.x - 1, coord.y, coord.z);
            auto [nr, nc] = vertical_neighbor(r, c);
            if (nr != -1) assign(flat_index(nr, nc), coord.x, coord.y - 1, coord.z);
        } else {
            if (c + 1 < ROW_LENGTHS[r]) assign(flat_index(r, c + 1), coord.x + 1, coord.y, coord.z);
            if (c - 1 >= 0) assign(flat_index(r, c - 1), coord.x, coord.y, coord.z + 1);
            auto [nr, nc] = vertical_neighbor(r, c);
            if (nr != -1) assign(flat_index(nr, nc), coord.x, coord.y + 1, coord.z);
        }
    }

    for (int i = 0; i < TOTAL_TRIANGLES; ++i) {
        COORD_TO_INDEX[INDEX_TO_COORD[i]] = i;
    }
}

// ----------------------------------------------------
// Mask Generation
static void generate_masks() {
    ALL_MASKS.clear();
    // 1. Horizontal
    for (int r = 0; r < 8; ++r) {
        BitBoard m;
        for (int c = 0; c < ROW_LENGTHS[r]; ++c) {
            m.set_bit(flat_index(r, c));
        }
        ALL_MASKS.push_back(m);
    }

    auto extract_lines = [&](auto next_fn) {
        std::vector<bool> visited(TOTAL_TRIANGLES, false);
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < ROW_LENGTHS[r]; ++c) {
                int idx = flat_index(r, c);
                if (visited[idx]) continue;

                int start_r = r, start_c = c;
                while (true) {
                    int prev_r = -1, prev_c = -1;
                    for (int pr = 0; pr < 8; ++pr) {
                        for (int pc = 0; pc < ROW_LENGTHS[pr]; ++pc) {
                            auto [nr, nc] = next_fn(pr, pc);
                            if (nr == start_r && nc == start_c) {
                                prev_r = pr; prev_c = pc;
                            }
                        }
                    }
                    if (prev_r == -1) break;
                    start_r = prev_r; start_c = prev_c;
                }

                BitBoard m;
                int curr_r = start_r, curr_c = start_c;
                while (curr_r != -1 && curr_c != -1) {
                    int i = flat_index(curr_r, curr_c);
                    m.set_bit(i);
                    visited[i] = true;
                    auto next_pos = next_fn(curr_r, curr_c);
                    curr_r = next_pos.first;
                    curr_c = next_pos.second;
                }
                ALL_MASKS.push_back(m);
            }
        }
    };

    auto next_red = [](int r, int c) -> std::pair<int, int> {
        if (is_up(r, c)) {
            if (c + 1 < ROW_LENGTHS[r]) return {r, c + 1};
        } else {
            auto nrnc = vertical_neighbor(r, c);
            if (nrnc.first != -1) return nrnc;
        }
        return {-1, -1};
    };

    auto next_black = [](int r, int c) -> std::pair<int, int> {
        if (is_up(r, c)) {
            if (c - 1 >= 0) return {r, c - 1};
        } else {
            auto nrnc = vertical_neighbor(r, c);
            if (nrnc.first != -1) return nrnc;
        }
        return {-1, -1};
    };

    extract_lines(next_red);
    extract_lines(next_black);
}

// ----------------------------------------------------
// Pieces Mapping
struct PieceDef {
    bool require_up;
    bool require_down;
    std::vector<Coord> offsets;
};

static void compile_pieces(const std::vector<PieceDef>& defs) {
    STANDARD_PIECES.clear();
    for (const auto& def_obj : defs) {
        std::vector<BitBoard> masks;
        masks.reserve(TOTAL_TRIANGLES);
        for (int i = 0; i < TOTAL_TRIANGLES; ++i) {
            bool is_upi = is_up_flat(i);
            if (def_obj.require_up && !is_upi) {
                masks.push_back(BitBoard());
                continue;
            }
            if (def_obj.require_down && is_upi) {
                masks.push_back(BitBoard());
                continue;
            }

            Coord origin = INDEX_TO_COORD[i];
            BitBoard m;
            bool valid = true;
            for (const auto& off : def_obj.offsets) {
                Coord t = {origin.x + off.x, origin.y + off.y, origin.z + off.z};
                if (COORD_TO_INDEX.find(t) == COORD_TO_INDEX.end()) {
                    valid = false;
                    break;
                }
                m.set_bit(COORD_TO_INDEX[t]);
            }
            masks.push_back(valid ? m : BitBoard());
        }
        STANDARD_PIECES.push_back(masks);
    }
}

void initialize_env() {
    build_coords();
    generate_masks();

    std::vector<PieceDef> defs = {
        {true, false, {{0,0,0}}},
        {false, true, {{0,0,0}}},
        {true, false, {{0,0,0}, {0,0,-1}, {1,0,-1}, {0,-1,0}, {1,-1,0}, {1,-1,-1}}},
        {true, false, {{0,0,0}, {0,0,-1}, {0,-1,0}, {1,-1,0}, {1,-1,-1}}},
        {true, false, {{0,0,0}, {0,0,-1}, {1,0,-1}, {0,-1,0}}},
        {true, false, {{0,0,0}, {0,-1,1}, {0,-1,0}, {1,-1,0}}},
        {false, true, {{0,0,0}, {1,0,0}, {1,0,-1}, {1,-1,0}}},
        {true, false, {{0,0,0}, {0,0,-1}, {1,0,-1}}},
        {false, true, {{0,0,0}, {1,0,0}, {1,0,-1}}},
        {true, false, {{0,0,0}, {0,0,-1}}},
        {true, false, {{0,0,0}, {0,-1,0}}},
        {true, false, {{0,0,0}, {0,-1,1}, {0,-1,0}}}
    };

    compile_pieces(defs);
}

// ----------------------------------------------------
// GameState Implementation

GameState::GameState() {
    board = BitBoard();
    score = 0;
    refill_tray();
    check_terminal();
}

GameState::GameState(std::vector<int> pieces, BitBoard board_state, int current_score) {
    available = pieces;
    board = board_state;
    score = current_score;
    pieces_left = 0;
    for (int p : available) {
        if (p != -1) pieces_left++;
    }
    check_terminal();
}

void GameState::check_terminal() {
    terminal = false;
    if (pieces_left > 0) {
        bool has_move = false;
        for (int p_id : available) {
            if (p_id == -1) continue;
            for (const auto& m : STANDARD_PIECES[p_id]) {
                if (!m.is_zero() && (board & m).is_zero()) {
                    has_move = true;
                    break;
                }
            }
            if (has_move) break;
        }
        terminal = !has_move;
    } else {
        terminal = true;
    }
}

GameState* GameState::apply_move(int slot, int index) {
    int p_id = available[slot];
    if (p_id == -1) return nullptr;

    BitBoard mask = STANDARD_PIECES[p_id][index];
    if (mask.is_zero() || !(board & mask).is_zero()) {
        return nullptr; // invalid move
    }

    GameState* next_state = new GameState(available, board, score);
    next_state->available[slot] = -1;
    next_state->pieces_left--;

    next_state->board |= mask;
    next_state->score += mask.count_ones(); // +1 point for each triangle on the piece/shape placed

    BitBoard cleared_mask(0, 0);
    int lines_cleared = 0;
    for (const auto& line : ALL_MASKS) {
        if ((next_state->board & line) == line) {
            cleared_mask |= line;
            lines_cleared++;
        }
    }

    if (lines_cleared > 0) {
        next_state->board &= ~cleared_mask;
        next_state->score += lines_cleared * 2; // +2 points for each triangle cleared by a line ? Or just +2 total per line?
        // User said: "+2 points for each triangle have being cleared by a line"
        // Meaning if a line of 9 triangles is cleared, is it +18 points?
        // Yes: "for each triangle have being cleared".
        // Let's implement EXACTLY what the user wrote:
        next_state->score += cleared_mask.count_ones() * 2; 
    }

    next_state->check_terminal();
    return next_state;
}

void GameState::refill_tray() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(0, 11);

    available = {distr(gen), distr(gen), distr(gen)};
    pieces_left = 3;
    check_terminal();
}

// Convert board to pure Python-friendly bit string string avoiding numeric overflow boundaries
std::string GameState::board_to_string() const {
    std::string s(TOTAL_TRIANGLES, '0');
    for (int i = 0; i < TOTAL_TRIANGLES; ++i) {
        if (board.get_bit(i)) {
            s[TOTAL_TRIANGLES - 1 - i] = '1'; // Python style, least significant bit at end
        }
    }
    return s;
}

BitBoard GameState::string_to_board(const std::string& s) {
    BitBoard b;
    int len = s.length();
    for (int i = 0; i < len; ++i) {
        if (s[len - 1 - i] == '1') {
            b.set_bit(i);
        }
    }
    return b;
}
