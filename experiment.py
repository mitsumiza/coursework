import numpy as np
from core import GraphCharacteristicPolynomial
from typing import List, Tuple, Dict
import itertools


class LatticeGraphExperiment:
    def __init__(self):
        self.calculator = GraphCharacteristicPolynomial()

    def generate_contour_points(self, a: int, b: int, c: int, d: int) -> List[Tuple[int, int]]:
        # Определяет контур (0,0)→(a,0)→(a,b)→(c,b)→(c,d)→(0,d)→(0,0) с условиями c < a и b < d.
        if c >= a or b >= d:
            raise ValueError("Некорректные параметры: должно быть c < a и b < d")

        points = set()

        # (0,0) -> (a,0)
        for x in range(0, a + 1):
            points.add((x, 0))

        # (a,0) -> (a,b)
        for y in range(0, b + 1):
            points.add((a, y))

        # (a,b) -> (c,b)
        for x in range(c, a + 1):
            points.add((x, b))

        # (c,b) -> (c,d)
        for y in range(b, d + 1):
            points.add((c, y))

        # (c,d) -> (0,d)
        for x in range(0, c + 1):
            points.add((x, d))

        # (0,d) -> (0,0)
        for y in range(0, d + 1):
            points.add((0, y))

        return sorted(list(points))

    def generate_interior_points(self, a: int, b: int, c: int, d: int) -> List[Tuple[int, int]]:

        # Генерирует белые точки (внутренние) внутри контура
        interior_points = []

        # Генерируем все точки внутри bounding box, исключая границы
        for x in range(1, max(a, c)):
            for y in range(1, max(b, d)):
                # Проверяем, что точка строго внутри контура
                if self._is_point_inside_contour((x, y), a, b, c, d):
                    interior_points.append((x, y))

        return interior_points

    def _is_point_inside_contour(self, point: Tuple[int, int], a: int, b: int, c: int, d: int) -> bool:

        # Проверяет, находится ли точка внутри L-образного контура

        x, y = point

        # Точка должна быть внутри внешнего прямоугольника (0,0)-(a,d)
        # но вне внутреннего прямоугольника (c,0)-(a,b)
        if x <= 0 or y <= 0 or x >= a or y >= d:
            return False

        # Исключаем точки, которые находятся в "вырезе" (c < x < a и 0 < y < b)
        if c < x < a and 0 < y < b:
            return False

        return True

    def count_interior_points(self, a: int, b: int, c: int, d: int) -> int:
        # Подсчитывает количество внутренних точек
        return len(self.generate_interior_points(a, b, c, d))

    def find_parameters_for_N(self, N: int, max_param: int = 8) -> List[Tuple[int, int, int, int]]:

        # Находит все комбинации параметров a,b,c,d дающие ровно N внутренних точек

        valid_parameters = []

        print(f"Поиск параметров для N={N} с max_param={max_param}...")

        for a in range(2, max_param + 1):
            for b in range(1, max_param + 1):
                for c in range(1, a):  # c должно быть меньше a
                    for d in range(b + 1, max_param + 1):  # d должно быть больше b
                        try:
                            interior_count = self.count_interior_points(a, b, c, d)
                            if interior_count == N:
                                valid_parameters.append((a, b, c, d))
                        except:
                            continue

        print(f"Найдено {len(valid_parameters)} наборов параметров")
        return valid_parameters

    def compute_graph_polynomial(self, a: int, b: int, c: int, d: int) -> Dict:

        # Вычисляет характеристический многочлен для графа с заданными параметрами

        try:
            # Генерируем вершины решетки (внутренние точки)
            vertices = self.generate_interior_points(a, b, c, d)

            if not vertices:
                return None

            # Создаем матрицу смежности
            adj_matrix = self.calculator.grid_adjacency_matrix(vertices)

            # Вычисляем характеристический многочлен
            poly_coeffs = self.calculator.characteristic_polynomial(adj_matrix)

            return {
                'parameters': (a, b, c, d),
                'coefficients': poly_coeffs,
                'n_vertices': len(vertices),
                'vertices': vertices
            }
        except Exception as e:
            print(f"Ошибка для параметров ({a},{b},{c},{d}): {e}")
            return None

    def run_experiment(self, N: int, max_param: int = 8) -> List[Dict]:

        # Проводит эксперимент для заданного N

        print(f"\n=== ЭКСПЕРИМЕНТ ДЛЯ N={N} ===")

        parameters_list = self.find_parameters_for_N(N, max_param)

        if not parameters_list:
            print(f"Не найдено параметров для N={N}")
            return []

        results = []
        print("Вычисление характеристических многочленов...")

        for i, params in enumerate(parameters_list):
            a, b, c, d = params
            result = self.compute_graph_polynomial(a, b, c, d)
            if result is not None:
                results.append(result)
                if len(results) % 10 == 0:
                    print(f"Обработано: {len(results)}/{len(parameters_list)}")

        print(f"Успешно вычислено: {len(results)} многочленов")
        return results

    def find_duplicate_polynomials(self, results: List[Dict]) -> Dict:

        # Находит графы с одинаковыми характеристическими многочленами

        poly_dict = {}

        for result in results:
            # Используем кортеж коэффициентов как ключ
            coeff_key = tuple(result['coefficients'])

            if coeff_key not in poly_dict:
                poly_dict[coeff_key] = []

            poly_dict[coeff_key].append(result)

        # Возвращаем только многочлены с несколькими наборами параметров
        duplicates = {k: v for k, v in poly_dict.items() if len(v) > 1}
        return duplicates


def save_experiment_results(results: List[Dict], filename: str):

    # Сохраняет результаты эксперимента в сжатый npz файл

    if not results:
        print("Нет результатов для сохранения")
        return

    # Подготавливаем данные для сохранения
    parameters = [result['parameters'] for result in results]
    coefficients = [result['coefficients'] for result in results]
    n_vertices = [result['n_vertices'] for result in results]

    # Сохраняем в сжатом формате
    np.savez_compressed(
        filename,
        parameters=parameters,
        coefficients=coefficients,
        n_vertices=n_vertices
    )
    print(f"Результаты сохранены в {filename}")


def load_experiment_results(filename: str) -> List[Dict]:

    # Загружает результаты эксперимента из файла

    data = np.load(filename, allow_pickle=True)

    results = []
    for i in range(len(data['parameters'])):
        results.append({
            'parameters': tuple(data['parameters'][i]),
            'coefficients': data['coefficients'][i],
            'n_vertices': data['n_vertices'][i]
        })

    return results


def analyze_duplicates(filename: str):

    # Анализирует файл с результатами на предмет дубликатов

    print(f"\nАНАЛИЗ ДУБЛИКАТОВ: {filename}")
    print("=" * 50)

    try:
        results = load_experiment_results(filename)
        experiment = LatticeGraphExperiment()

        duplicates = experiment.find_duplicate_polynomials(results)

        if duplicates:
            print(f"Найдено {len(duplicates)} многочленов с одинаковыми коэффициентами:")

            for i, (coeff, graphs) in enumerate(duplicates.items()):
                print(f"\nМногочлен #{i + 1}:")
                print(f"Коэффициенты: {coeff}")
                for graph in graphs:
                    a, b, c, d = graph['parameters']
                    print(f"  Параметры: a={a}, b={b}, c={c}, d={d}, вершин: {graph['n_vertices']}")
        else:
            print("Дубликаты не найдены")

    except FileNotFoundError:
        print(f"Файл {filename} не найден")
    except Exception as e:
        print(f"Ошибка при анализе: {e}")


def main():
    # Основная функция для запуска эксперимента
    experiment = LatticeGraphExperiment()

    # Простой тест
    print("ТЕСТИРОВАНИЕ ГЕНЕРАЦИИ ТОЧЕК")
    print("=" * 40)

    # Тестовый пример
    a, b, c, d = 4, 3, 2, 5
    print(f"Параметры: a={a}, b={b}, c={c}, d={d}")

    contour = experiment.generate_contour_points(a, b, c, d)
    interior = experiment.generate_interior_points(a, b, c, d)

    print(f"Точек контура: {len(contour)}")
    print(f"Внутренних точек: {len(interior)}")
    print(f"Внутренние точки: {interior}")

    # Запуск эксперимента для малого N
    N = 3
    results = experiment.run_experiment(N, max_param=6)

    if results:
        filename = f"experiment_N_{N}.npz"
        save_experiment_results(results, filename)
        analyze_duplicates(filename)
    else:
        print("Эксперимент не дал результатов")


if __name__ == "__main__":
    main()