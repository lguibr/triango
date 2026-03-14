#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "env.hpp"
#include "mcts.hpp"

namespace py = pybind11;

PYBIND11_MODULE(triango_ext, m) {
    m.doc() = "Triango completely native C++ environment engine";

    // Bind the global initialize function
    m.def("initialize_env", &initialize_env, "Initializes the C++ environment (must be called once)");

    // Expose the GameState class to Python
    py::class_<GameState>(m, "GameState")
        .def(py::init<>())
        .def(py::init([](std::vector<int> pieces, py::array_t<uint8_t> board_bytes, int score) {
            auto req = board_bytes.request();
            std::vector<uint8_t> vec((uint8_t*)req.ptr, ((uint8_t*)req.ptr) + req.size);
            return new GameState(pieces, GameState::bytes_to_board(vec), score);
        }))
        .def_readwrite("score", &GameState::score)
        .def_readwrite("available", &GameState::available)
        .def_readwrite("pieces_left", &GameState::pieces_left)
        .def_readwrite("terminal", &GameState::terminal)
        .def_property_readonly("board_bytes", [](const GameState& state) {
            auto vec = state.get_board_bytes();
            // Allocate true Python-owned numpy array so C++ vector lifecycle dies safely
            py::array_t<uint8_t> arr(vec.size());
            py::buffer_info r = arr.request(true);
            uint8_t* ptr = static_cast<uint8_t*>(r.ptr);
            std::copy(vec.begin(), vec.end(), ptr);
            return arr;
        })
        .def_property_readonly("board", [](const GameState& state) {
            // Backwards compatibility for python tests looking for an integer map
            // Note: BitBoard natively is uint64_t low & uint32_t hi.
            // Python tests only assume max 64 bit integer currently due to older architecture 
            // Better yet, just return the string since older python env relies on string/int mapping
            // But state.board expects an integer for mask masking.
            // We just return low bits.
            return static_cast<uint64_t>(state.board.to_int()); 
        })
        .def("check_terminal", &GameState::check_terminal)
        .def("get_valid_moves", [](const GameState& state) {
            std::vector<std::pair<int, int>> valid_moves;
            if (state.terminal) return valid_moves;
            for (int slot = 0; slot < 3; ++slot) {
                int p_id = state.available[slot];
                if (p_id == -1) continue;
                for (int idx = 0; idx < 96; ++idx) {
                    const BitBoard& mask = STANDARD_PIECES[p_id][idx];
                    if (!mask.is_zero() && (state.board & mask).is_zero()) {
                        valid_moves.push_back({slot, idx});
                    }
                }
            }
            return valid_moves;
        })
        .def("apply_move", [](GameState& self, int slot, int index) {
            GameState* next_state = self.apply_move(slot, index);
            return next_state; // pybind11 automatically handles the pointer return for Python
        }, py::return_value_policy::take_ownership) // Tell Python to take ownership of the newly allocated state
        .def("refill_tray", &GameState::refill_tray);

    py::class_<Node>(m, "Node")
        // The Python Node expects: Node(state, parent=None, move=None)
        // Since PyBind deals with pointers, we pass GameState* and Node*
        .def(py::init([](GameState* state, Node* parent, py::object move_obj) {
            std::pair<int, int> m = {-1, -1};
            if (!move_obj.is_none()) {
                m = move_obj.cast<std::pair<int, int>>();
            }
            // MCTS requires new independent Node trees.
            // Python passes a GameState it owns. We MUST clone it so Node owns its own copy and doesn't double-free the Python object!
            GameState* cloned_state = new GameState(*state);
            return new Node(cloned_state, parent, m);
        }), py::arg("state"), py::arg("parent") = nullptr, py::arg("move") = py::none())
        
        .def_readwrite("state", &Node::state, py::return_value_policy::reference)
        .def_readwrite("parent", &Node::parent, py::return_value_policy::reference)
        .def_property("visits", [](const Node& n) { return n.visits.load(); }, [](Node& n, int v) { n.visits.store(v); })
        .def_property("value_sum", [](const Node& n) { return n.value_sum.load(); }, [](Node& n, float v) { n.value_sum.store(v); })
        .def_readwrite("prior", &Node::prior)
        .def_property("virtual_loss", [](const Node& n) { return n.virtual_loss.load(); }, [](Node& n, int v) { n.virtual_loss.store(v); })
        .def_readwrite("expanded", &Node::expanded)
        .def_property_readonly("children", [](Node& n) {
            return n.children;
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("move", [](const Node& n) -> py::object {
            if (n.move.first == -1) return py::none();
            return py::cast(n.move);
        })
        .def("puct", &Node::puct, py::arg("c_puct") = 1.5f)
        .def("select_child", &Node::select_child, py::return_value_policy::reference_internal)
        .def("expand", &Node::expand, py::return_value_policy::reference_internal)
        .def("backpropagate", &Node::backpropagate, py::arg("reward"))
        .def("apply_dirichlet_noise", &Node::apply_dirichlet_noise, py::arg("alpha"), py::arg("epsilon"));


    // Bind EvalRequest so Python can read it
    py::class_<EvalRequest>(m, "EvalRequest")
        .def_readwrite("node", &EvalRequest::node, py::return_value_policy::reference)
        .def_readwrite("state", &EvalRequest::state);

    // Bind EvalResult so Python can create and pass it back
    py::class_<EvalResult>(m, "EvalResult")
        .def(py::init<Node*, float, std::vector<float>>(), 
             py::arg("node"), py::arg("value"), py::arg("policy"))
        .def_readwrite("node", &EvalResult::node, py::return_value_policy::reference)
        .def_readwrite("value", &EvalResult::value)
        .def_readwrite("policy", &EvalResult::policy);

    // Bind the Async Thread Pool Manager
    py::class_<AsyncMCTS>(m, "AsyncMCTS")
        .def(py::init<GameState*, int, int, float>(), 
             py::arg("root_state"), py::arg("threads") = 8, py::arg("sims") = 800, py::arg("c_puct") = 1.5f)
        .def("start", &AsyncMCTS::start)
        .def("stop", &AsyncMCTS::stop)
        .def("get_requests", &AsyncMCTS::get_requests, py::arg("max_batch") = 64)
        .def("submit_results", &AsyncMCTS::submit_results, py::arg("results"))
        .def("is_done", &AsyncMCTS::is_done)
        .def_readwrite("root", &AsyncMCTS::root, py::return_value_policy::reference);
}
