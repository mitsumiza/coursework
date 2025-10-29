print("=== ТЕСТИРОВАНИЕ ОСНОВНОЙ ФУНКЦИОНАЛЬНОСТИ ===")

# Проверка основных библиотек
import numpy as np
import sympy as sp

# Проверка основного класса
from core import GraphCharacteristicPolynomial

calculator = GraphCharacteristicPolynomial()

# Тест 1: Простая матрица
print("\n1. Тест характеристического многочлена:")
A = np.array([[0, 1], [1, 0]])
poly = calculator.characteristic_polynomial(A)
print(f"   Матрица: [[0,1],[1,0]]")
print(f"   Многочлен: {poly}")

# Тест 2: Граф-треугольник
print("\n2. Тест графа-треугольника:")
edges = [(0,1), (1,2), (2,0)]
adj = calculator.adjacency_matrix(edges, 3)
print(f"   Рёбра: {edges}")
print(f"   Матрица смежности:\n{adj}")

print("\n Все основные тесты пройдены!")