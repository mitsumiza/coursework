import numpy as np
from experiment import LatticeGraphExperiment, save_experiment_results, analyze_duplicates


def run_multiple_experiments():
    """Запускает эксперименты для разных значений N"""
    experiment = LatticeGraphExperiment()

    # Диапазон значений N для исследования
    N_values = [2, 3, 4, 5, 6]
    max_param = 8

    print("МАССИВНЫЙ ЭКСПЕРИМЕНТ")
    print("=" * 60)

    for N in N_values:
        print(f"\n>>> Исследование графов с {N} вершинами <<<")

        # Эксперимент
        results = experiment.run_experiment(N, max_param)

        if results:
            # Сохраняем результаты
            filename = f"experiment_results_N_{N}.npz"
            save_experiment_results(results, filename)

            # Анализируем дубликаты
            print(f"\nАнализ результатов для N={N}:")
            analyze_duplicates(filename)
        else:
            print(f"Для N={N} не найдено подходящих графов")


if __name__ == "__main__":
    run_multiple_experiments()