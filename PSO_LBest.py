import numpy as np
import matplotlib.pyplot as plt


class PSO:
    """Particle Swarm Optimization (minimization) with Euclidean LBest neighborhoods.

    Parameters
    ----------
    func : callable
        Function to minimize. Receives a 1D NumPy array of length ``dim``.
    bounds : array-like, shape (dim, 2)
        Lower and upper search bounds for each dimension.
    n_particles : int
        Number of particles in the swarm.
    max_iter : int
        Number of optimization iterations.
    w_start, w_end : float
        Inertia decreases linearly from ``w_start`` to ``w_end``.
    c1, c2 : float
        Cognitive and social coefficients.
    vmax_frac : float
        Maximum velocity as a fraction of each dimension's search range.
    seed : int or None
        Random seed. A fixed seed makes an individual run reproducible.
    keep_positions : bool
        Store particle positions at each iteration, for ``plot_swarm_2d``.
    n_neighbors : int
        Number of nearest particles in each particle's neighborhood, including
        the particle itself. Neighbors are recalculated on every iteration using
        Euclidean distance between their current positions. Must be between 1
        and ``n_particles``. If set to ``n_particles``, the method behaves as
        global-best PSO.

    Notes
    -----
    The global best is still tracked as ``best_position`` / ``best_value`` for
    reporting and convergence plots. It is NOT used to update particle
    velocities; the social component uses each particle's local best (LBest).
    """

    def __init__(self, func, bounds, n_particles=40, max_iter=200,
                 w_start=0.9, w_end=0.4, c1=1.5, c2=1.5,
                 vmax_frac=0.2, seed=None, keep_positions=False,
                 n_neighbors=5):
        self.func = func
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds debe tener forma (dim, 2).")
        if np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError("Cada límite inferior debe ser menor que su límite superior.")
        if n_particles < 1:
            raise ValueError("n_particles debe ser al menos 1.")
        if not 1 <= n_neighbors <= n_particles:
            raise ValueError("n_neighbors debe estar entre 1 y n_particles.")

        self.n_particles = int(n_particles)
        self.max_iter = int(max_iter)
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.vmax_frac = vmax_frac
        self.seed = seed
        self.keep_positions = keep_positions
        self.n_neighbors = int(n_neighbors)

        # Results (populated by run())
        self.best_position = None
        self.best_value = None
        self.history = {"best_val": [], "positions": []}
        self.experiment_results = None

    # Properties derived from bounds
    @property
    def dim(self):
        return len(self.bounds)

    @property
    def low(self):
        return self.bounds[:, 0]

    @property
    def high(self):
        return self.bounds[:, 1]

    # Algorithm
    def _evaluate(self, positions):
        return np.asarray([self.func(*p) for p in positions], dtype=float)

    def _initialize(self):
        """Create the swarm and reset its state at the start of each run."""
        self._rng = np.random.default_rng(self.seed)
        self._vmax = self.vmax_frac * (self.high - self.low)

        self.pos = self._rng.uniform(
            self.low, self.high, (self.n_particles, self.dim)
        )
        self.vel = self._rng.uniform(
            -self._vmax, self._vmax, (self.n_particles, self.dim)
        )

        self.pbest_pos = self.pos.copy()
        self.pbest_val = self._evaluate(self.pos)

        g = np.argmin(self.pbest_val)
        self.best_position = self.pbest_pos[g].copy()
        self.best_value = float(self.pbest_val[g])

        self.history = {
            "best_val": [self.best_value],
            "positions": [self.pos.copy()] if self.keep_positions else [],
        }

    def _inertia(self, it):
        frac = it / max(1, self.max_iter - 1)
        return self.w_start - (self.w_start - self.w_end) * frac

    def _get_local_best_positions(self):
        """Return one LBest position per particle using Euclidean KNN.

        The neighborhood is determined from the particles' CURRENT positions.
        For each particle, the best personal-best position among its K nearest
        particles (including itself) is selected as its local attractor.
        """
        # distances[i, j] = Euclidean distance from particle i to particle j
        differences = self.pos[:, np.newaxis, :] - self.pos[np.newaxis, :, :]
        distances = np.linalg.norm(differences, axis=2)

        # Indices of each particle's n_neighbors nearest particles.
        # argpartition avoids fully sorting every row.
        neighbors = np.argpartition(
            distances, kth=self.n_neighbors - 1, axis=1
        )[:, :self.n_neighbors]

        local_best_indices = np.empty(self.n_particles, dtype=int)
        for i, neighbor_indices in enumerate(neighbors):
            best_neighbor = np.argmin(self.pbest_val[neighbor_indices])
            local_best_indices[i] = neighbor_indices[best_neighbor]

        return self.pbest_pos[local_best_indices]

    def _step(self, it):
        """Perform one iteration using each particle's local best (LBest)."""
        w = self._inertia(it)
        r1 = self._rng.random((self.n_particles, self.dim))
        r2 = self._rng.random((self.n_particles, self.dim))

        # Local social attractor: best personal best among Euclidean neighbors.
        lbest_positions = self._get_local_best_positions()

        self.vel = (
            w * self.vel
            + self.c1 * r1 * (self.pbest_pos - self.pos)
            + self.c2 * r2 * (lbest_positions - self.pos)
        )
        self.vel = np.clip(self.vel, -self._vmax, self._vmax)
        self.pos = np.clip(self.pos + self.vel, self.low, self.high)

        vals = self._evaluate(self.pos)
        improved = vals < self.pbest_val
        self.pbest_pos[improved] = self.pos[improved]
        self.pbest_val[improved] = vals[improved]

        # Track the global best for reporting only; it does not drive movement.
        g = np.argmin(self.pbest_val)
        if self.pbest_val[g] < self.best_value:
            self.best_value = float(self.pbest_val[g])
            self.best_position = self.pbest_pos[g].copy()

        self.history["best_val"].append(self.best_value)
        if self.keep_positions:
            self.history["positions"].append(self.pos.copy())

    def run(self):
        """Execute one optimization run; return (best_position, best_value)."""
        self._initialize()
        for it in range(self.max_iter):
            self._step(it)
        return self.best_position.copy(), float(self.best_value)

    def run_multiple(self, n_runs=30):
        """Run several independent experiments and summarize their best fitness.

        A different random seed is generated for every run. The statistics are
        calculated from the final best fitness returned by each execution.

        Returns
        -------
        dict
            ``fitness_values``: best fitness from each run (NumPy array).
            ``best_fitness``: minimum value among all runs (minimization).
            ``worst_fitness``: maximum value among all runs.
            ``mean_fitness``: arithmetic mean of the run-best fitness values.
            ``std_fitness``: sample standard deviation (ddof=1); for a single
                run, population standard deviation (0.0) is returned.
            ``best_positions``: best position found in each run.
            ``run_results``: list of dictionaries with each run's position,
                fitness, and seed.

        The instance's ``seed`` is restored after the experiment batch. The
        optimizer's regular state (position/history) remains that of the last
        execution, as it does after calling ``run()``.
        """
        if not isinstance(n_runs, (int, np.integer)) or n_runs < 1:
            raise ValueError("n_runs debe ser un entero mayor o igual que 1.")

        original_seed = self.seed
        seed_sequence = np.random.SeedSequence(original_seed)
        child_sequences = seed_sequence.spawn(int(n_runs))
        run_results = []
        convergence_curves = []

        try:
            for run_index, child_sequence in enumerate(child_sequences, start=1):
                run_seed = int(child_sequence.generate_state(1)[0])
                self.seed = run_seed
                best_position, best_fitness = self.run()
                convergence_curves.append(list(self.history["best_val"]))
                run_results.append({
                    "run": run_index,
                    "seed": run_seed,
                    "best_position": best_position.copy(),
                    "best_fitness": float(best_fitness),
                })
        finally:
            self.seed = original_seed

        fitness_values = np.asarray(
            [result["best_fitness"] for result in run_results], dtype=float
        )
        best_run_index = int(np.argmin(fitness_values))
        worst_run_index = int(np.argmax(fitness_values))
        std_ddof = 1 if len(fitness_values) > 1 else 0

        summary = {
            "fitness_values": fitness_values,
            "best_fitness": float(np.min(fitness_values)),
            "worst_fitness": float(np.max(fitness_values)),
            "mean_fitness": float(np.mean(fitness_values)),
            "std_fitness": float(np.std(fitness_values, ddof=std_ddof)),
            "best_position": run_results[best_run_index]["best_position"].copy(),
            "worst_position": run_results[worst_run_index]["best_position"].copy(),
            "best_positions": np.asarray(
                [result["best_position"] for result in run_results]
            ),
            "run_results": run_results,
            "convergence_curves": np.asarray(convergence_curves, dtype=float), # (n_runs, max_iter + 1)
        }
        self.experiment_results = summary
        return summary

    # Plots
    def plot_convergence(self, save_path=None, show=True):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(self.history["best_val"])
        ax.set_xlabel("Iteración")
        ax.set_ylabel("Mejor valor encontrado")
        ax.set_title("Convergencia de PSO (LBest)")
        ax.set_yscale("symlog")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        return fig

    def plot_swarm_2d(self, frames=None, save_path=None, show=True):
        """Plot function contours and particle positions at selected iterations.

        Only works for 2D problems with ``keep_positions=True``.
        ``frames`` is a list of iteration indices; by default, four frames
        distributed across the saved history are selected.
        """
        if self.dim != 2:
            raise ValueError(
                "plot_swarm_2d solo funciona con problemas de 2 dimensiones."
            )
        if not self.history["positions"]:
            raise ValueError(
                "Crea el PSO con keep_positions=True y ejecuta run() primero."
            )

        n = len(self.history["positions"])
        if frames is None:
            frames = np.unique(np.linspace(0, n - 1, 4).astype(int))

        xs = np.linspace(*self.bounds[0], 200)
        ys = np.linspace(*self.bounds[1], 200)
        X, Y = np.meshgrid(xs, ys)
        Z = np.array([[self.func(x, y) for x in xs] for y in ys])

        fig, axes = plt.subplots(
            1, len(frames), figsize=(4 * len(frames), 4), sharey=True
        )
        for ax, frame in zip(np.atleast_1d(axes), frames):
            particles = self.history["positions"][frame]
            ax.contourf(X, Y, Z, levels=30, cmap="viridis")
            ax.scatter(
                particles[:, 0], particles[:, 1],
                c="white", edgecolors="k", s=25
            )
            ax.scatter(*self.best_position, c="red", marker="*", s=150)
            ax.set_title(f"Iteración {frame}")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        return fig
