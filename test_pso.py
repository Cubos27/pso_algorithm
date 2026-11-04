import numpy as np
import matplotlib.pyplot as plt
from PSO_LBest import PSO
import math_functions

# TARGET_FUNCTION = math_functions.rosenbrock
TARGET_FUNCTION = math_functions.michalewicz
BOUNDS = [(0, np.pi)] * 2 if TARGET_FUNCTION == math_functions.michalewicz else [(-5, 5)] * 2
N = 50
W = 0.9
C1 = 1.5
C2 = C1
name = TARGET_FUNCTION.__name__   # "rosenbrock" o "michalewicz"

pso = PSO(
    func=TARGET_FUNCTION,
    bounds=BOUNDS,
    max_iter=200,
    seed=55,
    keep_positions=True,
    n_neighbors=5,
    n_particles=N,
    c1=C1,
    c2=C2,
    w_start=0.9,
    w_end=0.4,
)

summary = pso.run_multiple(n_runs=30)

# Mostrar summary individuales
# print("\nFitness de cada ejecución:")
# print(summary["fitness_values"])

# Mostrar estadísticas
print(f"=== 30 Ejecuciones de {name} ===")
print(f"N: {N}")
print(f"w: {W}")
print(f"c1: {C1}, c2: {C2}\n")

print("Mejor Fitness:", summary["best_fitness"])
print("Peor Fitness:", summary["worst_fitness"])
print("Media:", summary["mean_fitness"])
print("Desviación Estándar:", summary["std_fitness"])

# ---------- Curva de convergencia promedio (30 ejecuciones) ----------
mean_curve = summary["convergence_curves"].mean(axis=0)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(mean_curve)
ax.set_xlabel("Iteración")
ax.set_ylabel("Mejor fitness promedio")
ax.set_title(f"Convergencia promedio (30 ejecuciones) - {name}")
if np.all(mean_curve > 0):
    ax.set_yscale("log")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"convergencia_{name}.png", dpi=150)

# ---------- Dispersion: inicial, intermedia y final ----------
# Una ejecucion representativa con semilla fija
pso.seed = 55
pso.run()

momentos = [("inicial", 0),
            ("intermedia", pso.max_iter // 2),
            ("final", pso.max_iter)]

for etiqueta, frame in momentos:
    fig_s = pso.plot_swarm_2d(frames=[frame],
                              save_path=f"scatter_{name}_{etiqueta}.png",
                              show=False)
    plt.close(fig_s)

plt.show()