"""
Characteristic Polynomial Calculator
Вычисление характеристических многочленов матриц графов
"""

from .core import GraphCharacteristicPolynomial
from .core import (
    print_polynomial, 
    print_matrix, 
    print_vertex_degrees,
    print_graph_info,
    print_section_title,
    print_result_section
)

__version__ = "1.0.0"
__author__ = "Ваше Имя"

__all__ = [
    'GraphCharacteristicPolynomial',
    'print_polynomial',
    'print_matrix', 
    'print_vertex_degrees',
    'print_graph_info',
    'print_section_title',
    'print_result_section'
]