import os
import numpy as np
import matplotlib.pyplot as plt

from ga_core import create_population, evolve_population


CHROMOSOME_LENGTH = 100
GENERATIONS = 100
RUNS = 20


def binary_to_decimal(individual):
    bits = "".join(str(bit) for bit in individual)
    return int(bits, 2)


def fitness(individual):
    return binary_to_decimal(individual)


def run_ga(pop_size=100, crossover_rate=0.7, mutation_rate=0.001):
    population = create_population(pop_size, CHROMOSOME_LENGTH)

    best_history = []
    avg_history = []

    for generation in range(1, GENERATIONS + 1):
        fitness_values = [fitness(ind) for ind in population]

        best_history.append(max(fitness_values))
        avg_history.append(np.mean(fitness_values))

        population = evolve_population(
            population=population,
            fitness_function=fitness,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            pop_size=pop_size
        )

    return {
        "best_history": best_history,
        "avg_history": avg_history
    }


def run_experiment(pop_size=100, crossover_rate=0.7, mutation_rate=0.001):
    all_best = []
    all_avg = []

    for _ in range(RUNS):
        result = run_ga(
            pop_size=pop_size,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate
        )

        all_best.append(result["best_history"])
        all_avg.append(result["avg_history"])

    return {
        "pop_size": pop_size,
        "crossover_rate": crossover_rate,
        "mutation_rate": mutation_rate,
        "best_histories": np.array(all_best),
        "avg_histories": np.array(all_avg)
    }


def plot_single_experiment(result, filename):
    mean_best = np.mean(result["best_histories"], axis=0)
    mean_avg = np.mean(result["avg_histories"], axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(mean_best, label="Best individual fitness")
    plt.plot(mean_avg, label="Average population fitness")

    plt.xlabel("Generation")
    plt.ylabel("Fitness / Decimal value")
    plt.title(
        f"GA 9.2 | n={result['pop_size']}, "
        f"Pc={result['crossover_rate']}, "
        f"Pm={result['mutation_rate']}"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_comparison(results, metric, title, filename):
    plt.figure(figsize=(10, 6))

    for result in results:
        histories = result[f"{metric}_histories"]
        mean_values = np.mean(histories, axis=0)

        label = (
            f"n={result['pop_size']}, "
            f"Pc={result['crossover_rate']}, "
            f"Pm={result['mutation_rate']}"
        )

        plt.plot(mean_values, label=label)

    plt.xlabel("Generation")
    plt.ylabel("Fitness / Decimal value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    base_result = run_experiment(
        pop_size=100,
        crossover_rate=0.7,
        mutation_rate=0.001
    )

    print("Running single experiment...")
    plot_single_experiment(
        base_result,
        "results/9_2/ga_9_2_base.png"
    )
    print("Single experiment complete.")

    print("Running population experiments...")
    population_experiments = [
        run_experiment(pop_size=50, crossover_rate=0.7, mutation_rate=0.001),
        run_experiment(pop_size=100, crossover_rate=0.7, mutation_rate=0.001),
        run_experiment(pop_size=200, crossover_rate=0.7, mutation_rate=0.001),
    ]
    print("Population experiments complete.")

    print("Crossover rate experiments running...")
    crossover_experiments = [
        run_experiment(pop_size=100, crossover_rate=0.0, mutation_rate=0.001),
        run_experiment(pop_size=100, crossover_rate=0.3, mutation_rate=0.001),
        run_experiment(pop_size=100, crossover_rate=0.7, mutation_rate=0.001),
        run_experiment(pop_size=100, crossover_rate=0.9, mutation_rate=0.001),
    ]
    print("Crossover rate experiments complete.")

    print("Mutation experiments running...")
    mutation_experiments = [
        run_experiment(pop_size=100, crossover_rate=0.7, mutation_rate=0.0001),
        run_experiment(pop_size=100, crossover_rate=0.7, mutation_rate=0.001),
        run_experiment(pop_size=100, crossover_rate=0.7, mutation_rate=0.005),
        run_experiment(pop_size=100, crossover_rate=0.7, mutation_rate=0.01),
    ]
    print("Mutation experiments complete.")

    plot_comparison(
        population_experiments,
        metric="best",
        title="Effect of Population Size on Best Individual",
        filename="results/ga_9_2_population_best.png"
    )

    plot_comparison(
        population_experiments,
        metric="avg",
        title="Effect of Population Size on Average Fitness",
        filename="results/9_2/ga_9_2_population_avg.png"
    )

    plot_comparison(
        crossover_experiments,
        metric="best",
        title="Effect of Crossover Rate on Best Individual",
        filename="results/9_2/ga_9_2_crossover_best.png"
    )

    plot_comparison(
        crossover_experiments,
        metric="avg",
        title="Effect of Crossover Rate on Average Fitness",
        filename="results/9_2/ga_9_2_crossover_avg.png"
    )

    plot_comparison(
        mutation_experiments,
        metric="best",
        title="Effect of Mutation Rate on Best Individual",
        filename="results/9_2/ga_9_2_mutation_best.png"
    )

    plot_comparison(
        mutation_experiments,
        metric="avg",
        title="Effect of Mutation Rate on Average Fitness",
        filename="results/9_2/ga_9_2_mutation_avg.png"
    )

    print("Exercise 9.2 completed.")
    print("Plots saved in the results/ folder.")
