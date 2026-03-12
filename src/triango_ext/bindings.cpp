#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
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
        .def(py::init([](std::vector<int> pieces, const std::string& board_str, int score) {
            return new GameState(pieces, GameState::string_to_board(board_str), score);
        }))
        .def_readwrite("score", &GameState::score)
        .def_readwrite("available", &GameState::available)
        .def_readwrite("pieces_left", &GameState::pieces_left)
        .def_readwrite("terminal", &GameState::terminal)
        .def_property_readonly("board", [](const GameState& state) {
            // Converts C++ BitBoard to a long string binary representation, then converts to Python int
            // to perfectly preserve precision
            std::string binaryStr = state.board_to_string();
            return py::module_::import("builtins").attr("int")(binaryStr, 2);
        })
        .def("check_terminal", &GameState::check_terminal)
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
            // But we must NOT delete the GameState passed by Python unless Python relinquishes it.
            return new Node(state, parent, m);
        }), py::arg("state"), py::arg("parent") = nullptr, py::arg("move") = py::none())
        
        .def_readwrite("state", &Node::state, py::return_value_policy::reference)
        .def_readwrite("parent", &Node::parent, py::return_value_policy::reference)
        .def_property("visits", [](const Node& n) { return n.visits.load(); }, [](Node& n, int v) { n.visits.store(v); })
        .def_property("value_sum", [](const Node& n) { return n.value_sum.load(); }, [](Node& n, float v) { n.value_sum.store(v); })
        .def_readwrite("prior", &Node::prior)
        .def_property("virtual_loss", [](const Node& n) { return n.virtual_loss.load(); }, [](Node& n, int v) { n.virtual_loss.store(v); })
        .def_readwrite("expanded", &Node::expanded)
        .def_readwrite("children", &Node::children, py::return_value_policy::reference)
        .def_property_readonly("move", [](const Node& n) -> py::object {
            if (n.move.first == -1) return py::none();
            return py::cast(n.move);
        })
        .def("puct", &Node::puct, py::arg("c_puct") = 1.5f)
        .def("select_child", &Node::select_child, py::return_value_policy::reference)
        .def("expand", &Node::expand, py::return_value_policy::reference)
        .def("backpropagate", &Node::backpropagate, py::arg("reward"));


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
