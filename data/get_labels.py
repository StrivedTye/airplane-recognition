import torch
import numpy as np


def soften(plane_type, v=0.8):
    type_dict = {0: np.array([1.0, v, 0.0, v, 0.0, 0.0, 0.0, 0.0, v, 0.0]),  # A320-200  C
                 1: np.array([v, 1.0, 0.0, v, 0.0, 0.0, 0.0, 0.0, v, 0.0]),  # B737-400  C
                 2: np.array([0.0, 0.0, 1.0, 0.0, v, v, 0.0, 0.0, 0.0, v]),  # A330-200  E
                 3: np.array([v, v, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, v, 0.0]),  # B737-800  C
                 4: np.array([0.0, 0.0, v, 0.0, 1.0, v, 0.0, 0.0, 0.0, v]),  # A340-200  E
                 5: np.array([0.0, 0.0, v, 0.0, v, 1.0, 0.0, 0.0, 0.0, v]),  # B747-200  E
                 6: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),  # B757-200  D
                 7: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),  # A380-800  F
                 8: np.array([v, v, 0.0, v, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),  # B737-200  C
                 9: np.array([0.0, 0.0, v, 0.0, v, v, 0.0, 0.0, 0.0, 1.0])}  # B777-200  E
    return type_dict[plane_type]
