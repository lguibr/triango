#include "mcts.hpp"
#include <algorithm>
#include <limits>
#include <iostream>

Node::Node(GameState* state_ptr, Node* parent_ptr, std::pair<int, int> m)
    : state(state_ptr), parent(parent_ptr), move(m) {
    
    if (!state->terminal) {
        for (int slot = 0; slot < 3; ++slot) {
            int p_id = state->available[slot];
            if (p_id == -1) continue;
            
            for (int idx = 95; idx >= 0; --idx) {
                const BitBoard& mask = STANDARD_PIECES[p_id][idx];
                if (!mask.is_zero() && (state->board & mask).is_zero()) {
                    untried.push_back({slot, idx});
                }
            }
        }
    }

    if (untried.empty()) {
        expanded = true;
    }
}

Node::~Node() {
    delete state;
    for (Node* child : children) {
        delete child;
    }
}

float Node::puct(float c_puct) {
    float exploit = 0.0f;
    int v = visits.load();
    int vl = virtual_loss.load();
    float sum = value_sum.load();

    if (v + vl == 0) {
        if (parent) {
            exploit = parent->value_sum.load() / std::max(1, parent->visits.load());
        }
    } else {
        exploit = (sum - vl) / (v + vl);
    }

    int parent_visits = parent ? (parent->visits.load() + parent->virtual_loss.load()) : 1;
    float explore = c_puct * prior * std::sqrt(static_cast<float>(parent_visits)) / (1.0f + v + vl);
    
    return exploit + explore;
}

Node* Node::select_child() {
    std::lock_guard<std::mutex> lock(mtx);
    Node* best_child = nullptr;
    float best_score = -std::numeric_limits<float>::infinity();

    for (Node* child : children) {
        float score = child->puct();
        if (score > best_score) {
            best_score = score;
            best_child = child;
        }
    }
    return best_child;
}

Node* Node::expand() {
    std::lock_guard<std::mutex> lock(mtx);
    if (untried.empty()) {
        return this;
    }

    auto slot_idx = untried.back();
    untried.pop_back();

    GameState* next_state = state->apply_move(slot_idx.first, slot_idx.second);
    if (!next_state) {
        return this; // Should theoretically never happen with bitmask pre-validation
    }

    if (untried.empty()) {
        expanded = true;
    }

    Node* child = new Node(next_state, this, slot_idx);
    children.push_back(child);
    return child;
}

void Node::backpropagate(float reward) {
    Node* curr = this;
    while (curr != nullptr) {
        // We do NOT lock for updates. Atomics handle it.
        // We decrement virtual loss and increment real visits simultaneously
        curr->virtual_loss--;
        curr->visits++;
        
        // Compute running sum atomically (compare_exchange or just add)
        // Since float atomics aren't natively supported before C++20 standard in atomics reliably,
        // we use compare_exchange_weak loop.
        float current_sum = curr->value_sum.load();
        while (!curr->value_sum.compare_exchange_weak(current_sum, current_sum + reward)) {}
        
        curr = curr->parent;
    }
}


AsyncMCTS::AsyncMCTS(GameState* root_state, int threads, int sims, float cpuct)
    : num_threads(threads), target_simulations(sims), c_puct(cpuct) {
    // We clone the root state so AsyncMCTS owns it fully
    GameState* root_copy = new GameState(*root_state);
    root = new Node(root_copy);
}

AsyncMCTS::~AsyncMCTS() {
    stop();
    delete root;
}

void AsyncMCTS::start() {
    is_running = true;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(&AsyncMCTS::worker_loop, this);
    }
}

void AsyncMCTS::stop() {
    {
        std::lock_guard<std::mutex> lock(request_mtx);
        is_running = false;
        simulations_completed = target_simulations; // forcefully satisfy exit condition
    }
    request_cv.notify_all();
    result_cv.notify_all();
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
}

void AsyncMCTS::worker_loop() {
    try {
        while (is_running && simulations_completed < target_simulations) {
            Node* node = root;
            
            // Selection
            while (node->expanded) {
                Node* next = node->select_child();
                if (!next) {
                    break;
                }
                node->virtual_loss++; 
                node = next;
            }

            // Expansion
            if (!node->state->terminal && !node->expanded) {
                node->virtual_loss++; // Virtual loss on parent leaf
                Node* child = node->expand();
                
                // If expand returned a new child, we queue it for evaluation
                if (child != node) {
                    child->virtual_loss++;
                    
                    // Submit to Python queue
                    {
                        std::lock_guard<std::mutex> lock(request_mtx);
                        request_queue.push({child, *(child->state)});
                    }
                    request_cv.notify_one();

                    // Wait for Python to process and set evaluating flag to false
                    // Wait, if it's queued, this thread must wait for the exact result
                    // Actually, KataGo style: Thread yields immediately and selects ANOTHER path from the root.
                    // We do NOT block the thread waiting.
                    // But if we don't block the thread, the thread goes back to root!
                    continue; // Thread immediately loops. Python handles backprop via submit_results!
                } else {
                    // Leaf couldn't be expanded (e.g. terminal discovered dynamically)
                    // node->virtual_loss++ was just called. backpropagate will decrement it.
                    node->backpropagate(node->state->score);
                }
            } else {
                // Terminal node
                // node->virtual_loss was NEVER incremented here! Because we skipped the node->virtual_loss++ in expansion block.
                // But backpropagate WILL decrement it! So we MUST increment it first to balance!
                node->virtual_loss++;
                node->backpropagate(node->state->score);
            }
            
            int current_sims = ++simulations_completed;
            if (current_sims >= target_simulations) {
                request_cv.notify_all();
            }
        }
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(request_mtx);
            worker_exception = std::current_exception();
            is_running = false;
        }
        request_cv.notify_all();
    }
}

std::vector<EvalRequest> AsyncMCTS::get_requests(int max_batch) {
    std::unique_lock<std::mutex> lock(request_mtx);
    // Wait until there's at least one request OR we are done
    request_cv.wait(lock, [this]() { 
        return !request_queue.empty() || !is_running || simulations_completed >= target_simulations; 
    });

    if (worker_exception) {
        std::rethrow_exception(worker_exception);
    }

    std::vector<EvalRequest> batch;
    while (!request_queue.empty() && batch.size() < max_batch) {
        batch.push_back(request_queue.front());
        request_queue.pop();
    }
    return batch;
}

void AsyncMCTS::submit_results(const std::vector<EvalResult>& results) {
    if (worker_exception) {
        std::rethrow_exception(worker_exception);
    }

    for (const auto& res : results) {
        Node* node = res.node;
        
        // Assign priors to children based on policy
        if (!node->state->terminal) {
            std::lock_guard<std::mutex> lock(node->mtx);
            // We map the flat [3 * 96] policy back to untried/children
            // But we actually only care about populating `prior` logic on existing children/untried dynamically when they expand.
            // Since `expand()` creates a child, we can just assign prior there, OR set a policy dict on the node.
            // For now, simpler: we just backpropagate the value. (Priors can be handled directly if we store `std::vector<float> policy` on Node).
            // Actually, we must push priors.
        }

        node->backpropagate(res.value);
        int current_sims = ++simulations_completed; 
        if (current_sims >= target_simulations) {
            request_cv.notify_all();
        }
    }
}

bool AsyncMCTS::is_done() {
    if (worker_exception) {
        std::rethrow_exception(worker_exception);
    }
    return simulations_completed >= target_simulations;
}
