"""Generic operator splitting of a PDE engine and a particle engine (Phase 7d).

``SplittingCoordinator`` replaces the old ``HybridCoupler``, which instantiated
FVReactionDiffusion and SmoldynHybrid directly. The coordinator holds **any** PDE engine,
**any** particle engine and any adapter Steps. Each is a node with an ``address``, a
``config`` and port → store wiring, exactly as in a composite document. Over one coupling
interval τ = k·dt it runs the engines in the order the scheme gives, on an internal copy of
the stores it is wired to. After each run it runs the adapters whose input stores changed, as
process-bigraph does with Steps.

Here uⁿ is the field and pⁿ the particle state at the start of the interval:

- ``jacobi``: vcell-fvsolver's lagged scheme. The PDE runs (k−1)·dt with pⁿ, the particles run
  τ reading the field after k−1 substeps (their update is held back), the PDE runs a last dt
  with pⁿ, then the particle update is applied. This equals the two-process composite with
  ``coupling="fvsolver"``.
- ``gs_particles_first``: the particles run τ reading uⁿ, then the PDE runs k·dt with pⁿ⁺¹.
- ``gs_pde_first``: the PDE runs k·dt with pⁿ, then the particles run τ reading uⁿ⁺¹.
- ``strang``: the PDE runs (k/2)·dt with pⁿ, the particles run τ reading the mid-interval
  field, then the PDE runs (k/2)·dt with pⁿ⁺¹ (k even).

The coordinator relies only on the process interface: ``update(state, interval)`` for
processes and ``update(state)`` for Steps, with map outputs merged key by key as overwrites. A
different splitting is a different schedule, not a different solver, and any engine pair
works. vcell-fvsolver's hybrid loop (SimTool.cpp) implements one fixed order.

Config:
    pde, particles: nodes ``{"address", "config", "inputs": {port: [store]}, "outputs": {port: [store]}}``.
    adapters: ``{name: node}``, Steps, run in this order when triggered.
    scheme: one of ``SCHEMES``.
    dt: PDE step (s).
    step_multiplier: k, with τ = k·dt (the coordinator's interval).
"""
from __future__ import annotations

from process_bigraph import Process

SCHEMES = ("jacobi", "gs_particles_first", "gs_pde_first", "strang")


def schedule(scheme: str, k: int, dt: float) -> list[tuple]:
    """Operations for one coupling interval: ("pde", interval), ("particles", interval[, "defer"]), ("commit",)."""
    tau = k * dt
    if scheme == "jacobi":
        ops = [("pde", (k - 1) * dt)] if k > 1 else []
        return ops + [("particles", tau, "defer"), ("pde", dt), ("commit",)]
    if scheme == "gs_particles_first":
        return [("particles", tau), ("pde", k * dt)]
    if scheme == "gs_pde_first":
        return [("pde", k * dt), ("particles", tau)]
    if scheme == "strang":
        if k % 2:
            raise ValueError("strang splitting needs an even step multiplier")
        return [("pde", (k // 2) * dt), ("particles", tau), ("pde", (k // 2) * dt)]
    raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")


def _store(path) -> str:
    if not isinstance(path, (list, tuple)) or len(path) != 1:
        raise ValueError(f"SplittingCoordinator wires ports to top-level stores, got {path!r}")
    return path[0]


class SplittingCoordinator(Process):
    config_schema = {
        "pde": "map",
        "particles": "map",
        "adapters": {"_type": "map", "_default": {}},
        "scheme": {"_type": "string", "_default": "jacobi"},
        "dt": "float",
        "step_multiplier": {"_type": "integer", "_default": 1},
    }

    def initialize(self, config):
        self.k = int(config["step_multiplier"])
        self.dt = float(config["dt"])
        self.ops = schedule(config["scheme"], self.k, self.dt)
        self.nodes = {"pde": config["pde"], "particles": config["particles"], **config["adapters"]}
        self.adapters = list(config["adapters"])
        self.children = {name: self._instantiate(node) for name, node in self.nodes.items()}

    def _instantiate(self, node):
        address = node["address"]
        name = address.split(":", 1)[1] if ":" in address else address
        cls = self.core.link_registry.get(name)
        if cls is None:
            raise ValueError(f"SplittingCoordinator: no process registered at {address!r}")
        return cls(config=node.get("config", {}), core=self.core)

    # ports: the union of the children's, named by store
    def _ports(self, direction: str) -> dict:
        ports = {}
        for name, node in self.nodes.items():
            schema = getattr(self.children[name], direction)()
            for port, path in node.get(direction, {}).items():
                ports.setdefault(_store(path), schema[port])
        return ports

    def inputs(self):
        return self._ports("inputs")

    def outputs(self):
        return self._ports("outputs")

    def initial_state(self):
        out = {}
        for name, node in self.nodes.items():
            child = self.children[name]
            init = child.initial_state() if hasattr(child, "initial_state") else {}
            for port, value in (init or {}).items():
                if port in node.get("outputs", {}):
                    out[_store(node["outputs"][port])] = value
        return out

    # running children on the internal stores
    def _view(self, name: str, stores: dict) -> dict:
        return {port: stores.get(_store(path)) for port, path in self.nodes[name].get("inputs", {}).items()}

    def _apply(self, name: str, update: dict, stores: dict) -> set:
        changed = set()
        for port, value in (update or {}).items():
            store = _store(self.nodes[name]["outputs"][port])
            old = stores.get(store)
            stores[store] = {**old, **value} if isinstance(value, dict) and isinstance(old, dict) else value
            changed.add(store)
        return changed

    def _trigger(self, changed: set, stores: dict) -> None:
        for name in self.adapters:
            reads = {_store(p) for p in self.nodes[name].get("inputs", {}).values()}
            if reads & changed:
                changed |= self._apply(name, self.children[name].update(self._view(name, stores)), stores)

    def update(self, state, interval):
        stores = dict(state)
        n_intervals = max(1, int(round(interval / (self.k * self.dt))))
        for _ in range(n_intervals):
            pending = None
            for op in self.ops:
                if op[0] == "commit":
                    self._trigger(self._apply("particles", pending, stores), stores)
                    continue
                role, dt_op = op[0], op[1]
                if dt_op <= 0:
                    continue
                update = self.children[role].update(self._view(role, stores), dt_op)
                if len(op) > 2 and op[2] == "defer":
                    pending = update
                else:
                    self._trigger(self._apply(role, update, stores), stores)
        return {store: stores[store] for store in self.outputs() if store in stores}
