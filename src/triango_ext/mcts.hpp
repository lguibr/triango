#pragma once

#include "env.hpp"
#include <vector>
#include <cmath>
#include <utility>
#include <exception>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <queue>
#include <memory>
#include <thread>

struct Node {
    GameState* state;
    Node* parent;
    std::pair<int, int> move; // {slot, index}
    
    std::atomic<int> visits{0};
    std::atomic<float> value_sum{0.0f};
    float prior = 0.0f;
    std::atomic<int> virtual_loss{0};
    
    bool expanded = false;
    bool is_evaluating = false;
    
    std::mutex mtx; // Protects children and untried

    std::vector<Node*> children;
    std::vector<std::pair<int, int>> untried;

    Node(GameState* state_ptr, Node* parent_ptr = nullptr, std::pair<int, int> m = {-1, -1});
    ~Node();

    float puct(float c_puct = 1.5f);
    Node* select_child();
    Node* expand();
    void backpropagate(float reward);
};

struct EvalRequest {
    Node* node;
    GameState state; // Copy for python to evaluate
};

struct EvalResult {
    Node* node;
    float value;
    std::vector<float> policy; // Flattened policy matches PyTorch output [3 * 96]
};

class AsyncMCTS {
public:
    Node* root;
    int num_threads;
    std::atomic<int> simulations_completed{0};
    int target_simulations;
    float c_puct;

    // Search thread pool
    std::vector<std::thread> threads;
    std::atomic<bool> is_running{false};

    // Queue for requests going FROM C++ TO Python
    std::queue<EvalRequest> request_queue;
    std::mutex request_mtx;
    std::condition_variable request_cv;

    // Queue for results going FROM Python TO C++
    std::queue<EvalResult> result_queue;
    std::mutex result_mtx;
    std::condition_variable result_cv;

    std::exception_ptr worker_exception = nullptr;

    AsyncMCTS(GameState* root_state, int threads = 8, int sims = 800, float c_puct = 1.5f);
    ~AsyncMCTS();

    void start();
    void stop();
    void worker_loop();

    // Python interacting functions
    std::vector<EvalRequest> get_requests(int max_batch);
    void submit_results(const std::vector<EvalResult>& results);
    bool is_done();
};
