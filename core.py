import numpy as np
import sympy as sp
from typing import List, Set, Tuple, Dict


class GraphCharacteristicPolynomial:
    # Класс для вычисления характеристических многочленов матриц графов

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

    if len(coeffs) == 0:
        print(f"{name} = 0")
        return


    if coeffs[0] < 0:
        coeffs = -coeffs

    n = len(coeffs) - 1
    terms = []

    for i, coeff in enumerate(coeffs):
        power = n - i
        if coeff == 0:
            continue


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


    poly_str = " + ".join(terms).replace("+ -", "- ")
    print(f"{name} = {poly_str}")


def print_matrix(matrix: np.ndarray, name: str = "Матрица"):

    print(f"{name}:")
    print(matrix)
    print()


def print_vertex_degrees(degrees: List[int], name: str = "Степени вершин"):

    print(f"{name}:")
    for i, deg in enumerate(degrees):
        print(f"  Вершина {i}: степень {deg}")
    print()


def print_graph_info(edges: List[Tuple[int, int]], n_vertices: int, graph_name: str = "Граф"):

    print(f"{graph_name}:")
    print(f"   Количество вершин: {n_vertices}")
    print(f"   Количество ребер: {len(edges)}")
    print(f"   Рёбра: {edges}")
    print()


def print_section_title(title: str, width: int = 60):

    print("=" * width)
    print(f"{title.upper()}")
    print("=" * width)
    print()


def print_result_section(title: str):

    print(f"\n{title}:")
    print("-" * 50)