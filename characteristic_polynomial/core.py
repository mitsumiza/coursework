import numpy as np
import sympy as sp
from typing import List, Set, Tuple, Dict


class GraphCharacteristicPolynomial:
    # Класс для вычисления характеристических многочленов матриц графов с использованием SymPy и NumPy

    @staticmethod
    def characteristic_polynomial(matrix: np.ndarray) -> np.ndarray:
        # Вычисляет характеристический многочлен целочисленной матрицы
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Матрица должна быть квадратной")

        n = matrix.shape[0]
        lam = sp.symbols('λ')
        A = sp.Matrix(matrix)  # Преобразуем NumPy массив в SymPy матрицу
        E = sp.eye(n)
        char_matrix = A - lam * E
        poly = char_matrix.det()

        # Преобразуем к полиному и извлекаем коэффициенты
        poly_expr = sp.Poly(poly, lam)
        coeffs = poly_expr.all_coeffs()

        # Преобразуем коэффициенты к целым числам и возвращаем как NumPy массив
        return np.array([int(coeff) for coeff in coeffs], dtype=int)

    @staticmethod
    def adjacency_matrix(edges: List[Tuple[int, int]], n_vertices: int) -> np.ndarray:
        # Создает матрицу смежности графа как NumPy массив
        adj = np.zeros((n_vertices, n_vertices), dtype=int)

        for i, j in edges:
            if i >= n_vertices or j >= n_vertices:
                raise ValueError(f"Вершины должны быть в диапазоне [0, {n_vertices - 1}]")
            adj[i, j] = 1
            adj[j, i] = 1  # Для неориентированного графа

        return adj

    @staticmethod
    def laplacian_matrix(adj_matrix: np.ndarray) -> np.ndarray:
        # Создает матрицу Кирхгофа графа как NumPy массив
        n = adj_matrix.shape[0]
        degrees = np.diag(np.sum(adj_matrix, axis=1))
        return degrees - adj_matrix

    @staticmethod
    def remove_vertices(matrix: np.ndarray, vertices_to_remove: Set[int]) -> np.ndarray:
        # Удаляет строки и столбцы, соответствующие заданным вершинам
        keep_vertices = [i for i in range(matrix.shape[0]) if i not in vertices_to_remove]
        return matrix[np.ix_(keep_vertices, keep_vertices)]

    @staticmethod
    def grid_adjacency_matrix(vertices: List[Tuple[int, int]]) -> np.ndarray:
        # Создает матрицу смежности для подграфа целочисленной решетки как NumPy массив
        n = len(vertices)
        adj = np.zeros((n, n), dtype=int)
        coord_to_idx = {coord: i for i, coord in enumerate(vertices)}

        for i, (x1, y1) in enumerate(vertices):
            for dx, dy in [(1, 0), (0, 1)]:  # Проверяем соседей справа и сверху
                neighbor = (x1 + dx, y1 + dy)
                if neighbor in coord_to_idx:
                    j = coord_to_idx[neighbor]
                    adj[i, j] = 1
                    adj[j, i] = 1
        return adj

    def compute_all_polynomials(self, edges: List[Tuple[int, int]],
                                n_vertices: int,
                                vertices_to_remove: Set[int] = None) -> Dict[str, np.ndarray]:
        # Вычисляет все требуемые характеристические многочлены для заданного графа
        if vertices_to_remove is None:
            vertices_to_remove = set()

        # Создаем матрицы
        adj = self.adjacency_matrix(edges, n_vertices)
        laplacian = self.laplacian_matrix(adj)

        # Вычисляем характеристические многочлены
        result = {
            'adjacency_poly': self.characteristic_polynomial(adj),
            'laplacian_poly': self.characteristic_polynomial(laplacian)
        }

        # Если заданы вершины для удаления
        if vertices_to_remove:
            adj_reduced = self.remove_vertices(adj, vertices_to_remove)
            laplacian_reduced = self.remove_vertices(laplacian, vertices_to_remove)

            result['adjacency_reduced_poly'] = self.characteristic_polynomial(adj_reduced)
            result['laplacian_reduced_poly'] = self.characteristic_polynomial(laplacian_reduced)

        return result

    @staticmethod
    def get_vertex_degrees(adj_matrix: np.ndarray) -> List[int]:
        # Вычисляет степени вершин графа
        return [int(np.sum(adj_matrix[i])) for i in range(adj_matrix.shape[0])]


# Функции для вывода
def print_polynomial(coeffs: np.ndarray, name: str = "P(λ)"):
    # Красиво выводит многочлен по его коэффициентам
    if len(coeffs) == 0:
        print(f"{name} = 0")
        return

    # Нормализуем: делаем старший коэффициент положительным
    if coeffs[0] < 0:
        coeffs = -coeffs

    n = len(coeffs) - 1
    terms = []

    for i, coeff in enumerate(coeffs):
        power = n - i
        if coeff == 0:
            continue

        # Форматируем коэффициент
        if power == 0:
            term = f"{coeff}"
        elif power == 1:
            if coeff == 1:
                term = "λ"
            elif coeff == -1:
                term = "-λ"
            else:
                term = f"{coeff}λ"
        else:
            if coeff == 1:
                term = f"λ^{power}"
            elif coeff == -1:
                term = f"-λ^{power}"
            else:
                term = f"{coeff}λ^{power}"

        terms.append(term)

    # Собираем многочлен
    poly_str = " + ".join(terms).replace("+ -", "- ")
    print(f"{name} = {poly_str}")


def print_matrix(matrix: np.ndarray, name: str = "Матрица"):
    # Выводит матрицу в читаемом формате
    print(f"{name}:")
    print(matrix)
    print()


def print_vertex_degrees(degrees: List[int], name: str = "Степени вершин"):
    # Красиво выводит степени вершин графа
    print(f"{name}:")
    for i, deg in enumerate(degrees):
        print(f"  Вершина {i}: степень {deg}")
    print()


def print_graph_info(edges: List[Tuple[int, int]], n_vertices: int, graph_name: str = "Граф"):
    # Выводит информацию о графе в красивом формате
    print(f"{graph_name}:")
    print(f"   Количество вершин: {n_vertices}")
    print(f"   Количество ребер: {len(edges)}")
    print(f"   Рёбра: {edges}")
    print()


def print_section_title(title: str, width: int = 60):
    # Выводит заголовок секции в красивом формате
    print("=" * width)
    print(f"{title.upper()}")
    print("=" * width)
    print()


def print_result_section(title: str):
    # Выводит заголовок для раздела с результатами
    print(f"\n{title}:")
    print("-" * 50)


# Пример использования
if __name__ == "__main__":
    calculator = GraphCharacteristicPolynomial()

    print_section_title("ВЫЧИСЛЕНИЕ ХАРАКТЕРИСТИЧЕСКИХ МНОГОЧЛЕНОВ")

    # Пример 1: Произвольная целочисленная матрица
    print_result_section("ПРОИЗВОЛЬНАЯ ЦЕЛОЧИСЛЕННАЯ МАТРИЦА")
    A = np.array([[2, 1], [1, 3]])
    poly_A = calculator.characteristic_polynomial(A)
    print_matrix(A, "Матрица A")
    print_polynomial(poly_A, "P_A(λ)")

    # Пример 2: Граф-треугольник
    print_result_section("ГРАФ-ТРЕУГОЛЬНИК (K3)")
    edges_triangle = [(0, 1), (1, 2), (2, 0)]
    n_vertices = 3

    print_graph_info(edges_triangle, n_vertices, "Граф-треугольник")
    result = calculator.compute_all_polynomials(edges_triangle, n_vertices)

    # Матрица смежности
    adj = calculator.adjacency_matrix(edges_triangle, n_vertices)
    print_matrix(adj, "Матрица смежности")
    print_polynomial(result['adjacency_poly'], "P_adj(λ)")

    # Степени вершин
    degrees = calculator.get_vertex_degrees(adj)
    print_vertex_degrees(degrees)

    # Матрица Кирхгофа
    laplacian = calculator.laplacian_matrix(adj)
    print_matrix(laplacian, "Матрица Кирхгофа")
    print_polynomial(result['laplacian_poly'], "P_lapl(λ)")

    # Пример 3: Удаление вершины
    print_result_section("УДАЛЕНИЕ ВЕРШИНЫ ИЗ ГРАФА-ТРЕУГОЛЬНИКА")
    vertices_to_remove = {0}
    result_reduced = calculator.compute_all_polynomials(edges_triangle, n_vertices, vertices_to_remove)

    adj_reduced = calculator.remove_vertices(adj, vertices_to_remove)
    print(f"Удалены вершины: {vertices_to_remove}")
    print_matrix(adj_reduced, "Матрица смежности после удаления вершин")
    print_polynomial(result_reduced['adjacency_reduced_poly'], "P_adj_reduced(λ)")

    # Пример 4: Граф целочисленной решетки
    print_result_section("ПОДГРАФ ЦЕЛОЧИСЛЕННОЙ РЕШЕТКИ (КВАДРАТ 2X2)")
    grid_vertices = [(0, 0), (0, 1), (1, 0), (1, 1)]
    grid_adj = calculator.grid_adjacency_matrix(grid_vertices)
    print(f"Вершины решетки: {grid_vertices}")
    print_matrix(grid_adj, "Матрица смежности решетки")

    grid_poly = calculator.characteristic_polynomial(grid_adj)
    print_polynomial(grid_poly, "P_grid(λ)")

    # Пример 5: Граф со степенью вершин ≤ 4 (простой путь)
    print_result_section("ГРАФ СО СТЕПЕНЬЮ ВЕРШИН ≤ 4 (ПУТЬ ИЗ 4 ВЕРШИН)")
    edges_path = [(0, 1), (1, 2), (2, 3)]
    n_path = 4

    print_graph_info(edges_path, n_path, "Путь P4")
    result_path = calculator.compute_all_polynomials(edges_path, n_path)

    adj_path = calculator.adjacency_matrix(edges_path, n_path)
    print_matrix(adj_path, "Матрица смежности пути")

    degrees_path = calculator.get_vertex_degrees(adj_path)
    print_vertex_degrees(degrees_path)
    print_polynomial(result_path['adjacency_poly'], "P_path(λ)")

    # Пример 6: Полный граф K4
    print_result_section("ПОЛНЫЙ ГРАФ K4")
    edges_k4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    n_k4 = 4

    print_graph_info(edges_k4, n_k4, "Полный граф K4")
    result_k4 = calculator.compute_all_polynomials(edges_k4, n_k4, {3})

    adj_k4 = calculator.adjacency_matrix(edges_k4, n_k4)
    print_matrix(adj_k4, "Матрица смежности K4")
    print_polynomial(result_k4['adjacency_poly'], "P_adj(K4)")

    degrees_k4 = calculator.get_vertex_degrees(adj_k4)
    print_vertex_degrees(degrees_k4)

    laplacian_k4 = calculator.laplacian_matrix(adj_k4)
    print_matrix(laplacian_k4, "Матрица Кирхгофа K4")
    print_polynomial(result_k4['laplacian_poly'], "P_lapl(K4)")

    print("\nПосле удаления вершины 3:")
    adj_k4_reduced = calculator.remove_vertices(adj_k4, {3})
    print_matrix(adj_k4_reduced, "Матрица смежности K4 без вершины 3")
    print_polynomial(result_k4['adjacency_reduced_poly'], "P_adj(K4 без вершины 3)")

    print("\n" + "ВЫЧИСЛЕНИЯ ЗАВЕРШЕНЫ!".center(60, "="))