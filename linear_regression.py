from __future__ import annotations

from typing import List

import numpy as np

from descents import BaseDescent
from descents import get_descent


class LinearRegression:
    """
    Класс линейной регрессии.

    Parameters
    ----------
    descent_config : dict
        Конфигурация градиентного спуска.
    tolerance : float, optional
        Критерий остановки для квадрата евклидова нормы разности весов. По умолчанию равен 1e-4.
    max_iter : int, optional
        Критерий остановки по количеству итераций. По умолчанию равен 300.

    Attributes
    ----------
    descent : BaseDescent
        Экземпляр класса, реализующего градиентный спуск.
    tolerance : float
        Критерий остановки для квадрата евклидова нормы разности весов.
    max_iter : int
        Критерий остановки по количеству итераций.
    loss_history : List[float]
        История значений функции потерь на каждой итерации.

    """

    def __init__(self, descent_config: dict, tolerance: float = 1e-4, max_iter: int = 300):
        """
        :param descent_config: gradient descent config
        :param tolerance: stopping criterion for square of euclidean norm of weight difference (float)
        :param max_iter: stopping criterion for iterations (int)
        """
        self.descent: BaseDescent = get_descent(descent_config)

        self.tolerance: float = tolerance
        self.max_iter: int = max_iter

        self.loss_history: List[float] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearRegression:
        """
        Обучение модели линейной регрессии, подбор весов для наборов данных x и y.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        self : LinearRegression
            Возвращает экземпляр класса с обученными весами.

        """
        # TODO: реализовать подбор весов для x и y
        self.loss_history.append(self.calc_loss(x, y))  # функции потерь до обучения

        for i in range(1, self.max_iter + 1):
            weight_difference = self.descent.step(x, y)  # обновление весов
            current_loss = self.calc_loss(x, y)
            self.loss_history.append(current_loss)  # текущая функция потерь
            if np.linalg.norm(weight_difference) < self.tolerance: # Евклидова норма разности векторов весов между двумя последовательными итерациями становится меньше заданного порога
                print(f"Сходимость на {i+1} итерации")
                break

            if np.isnan(current_loss) or np.any(np.isnan(self.descent.w)): # В векторе весов появляются значения NaN
                print("Обнаружен NaN в векторе весов")
                break

        return self


        raise NotImplementedError('Функция fit класса LinearRegression не реализована')

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Прогнозирование целевых переменных для набора данных x.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.

        Returns
        -------
        prediction : np.ndarray
            Массив прогнозируемых значений.
        """
        return self.descent.predict(x)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Вычисление коэффициента детерминации R^2.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        r2 : float
            Значение коэффициента детерминации R^2.
        """
        y_pred = self.predict(x)
        chisl = np.sum((y - y_pred) ** 2)
        znamen = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (chisl / znamen)
        return r2

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Расчёт значения функции потерь для наборов данных x и y.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        loss : float
            Значение функции потерь.
        """
        return self.descent.calc_loss(x, y)
