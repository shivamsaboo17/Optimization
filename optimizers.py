import numpy as np
import matplotlib.pyplot as plt


class SteepestDescent:
    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.x0 = np.random.randint(0, 10, b.shape)
        self.xs = [self.x0]
    def get_residual(self, x):
        return self.b - np.transpose(np.matmul(self.A, np.transpose(x)))
    def line_search(self, residual):
        return np.dot(residual, residual) / np.dot(np.matmul(residual, self.A), residual)
    def update(self, x, line, residual):
        return x + line * residual
    def converge(self, iterations=5):
        for _ in range(iterations):
            residual = self.get_residual(self.x0)
            line = self.line_search(residual)
            self.x0 = self.update(self.x0, line, residual)
            self.xs.append(self.x0)
        self.xs = np.array(self.xs)
        return self.x0
    def plot_convergence(self):
        def fx(x, y):
            return (1/2) * (self.A[0, 0] * x ** 2 + self.A[1, 1] * y ** 2 + 2 * self.A[0, 1] * x * y) - self.b[0] * x - self.b[1] * y
        xvalues = np.linspace(-np.max(self.xs[0]), np.max(self.xs[0]), int(abs(2 * np.max(self.xs[0]))))
        yvalues = np.linspace(-np.max(self.xs[0]), np.max(self.xs[0]), int(abs(2 * np.max(self.xs[0]))))
        x, y = np.meshgrid(xvalues, yvalues)
        z = fx(x, y)
        plt.contour(x, y, z)
        plt.plot(self.xs[:, 0], self.xs[:, 1])


class ConjugateDirection:
    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.x0 = self.x0 = np.random.randint(0, 10, b.shape)
        self.U = np.array([[0, 1], [1, 0]])
        self.d_list = [self.U[0]]
        self.beta_dict = {}
        self.xs = [self.x0]
    def get_residual(self, x):
        return self.b - np.transpose(np.matmul(self.A, np.transpose(x)))
    def update_orth_direction_list(self, i):
        _component_sum = np.zeros(self.b.shape)
        for k in range(i):
            if (i, k) not in self.beta_dict:
                self.update_beta_dict((i, k))
            _component_sum += self.beta_dict[(i, k)] * self.d_list[k]
        self.d_list.append(self.U[i] + _component_sum)
    def update_beta_dict(self, key):
        u, d = self.U[key[0]], self.d_list[key[1]]
        self.beta_dict[key] = -np.dot(np.matmul(u, self.A), d) / np.dot(np.matmul(d, self.A), d)
    def line_search(self, residual, direction):
        return np.dot(direction, residual) / np.dot(np.matmul(direction, self.A), direction)
    def update(self, x, line, direction):
        return x + line * direction
    def converge(self):
        iterations = self.A.shape[0]
        for i in range(iterations):
            residual = self.get_residual(self.x0)
            if i > 0:
                self.update_orth_direction_list(i)
            line = self.line_search(residual, self.d_list[i])
            self.x0 = self.update(self.x0, line, self.d_list[i])
            self.xs.append(self.x0)
        self.xs = np.array(self.xs)
        return self.x0
    def plot_convergence(self):
        def fx(x, y):
            return (1/2) * (self.A[0, 0] * x ** 2 + self.A[1, 1] * y ** 2 + 2 * self.A[0, 1] * x * y) - self.b[0] * x - self.b[1] * y
        xvalues = np.linspace(-6, 6, int(abs(6 * np.max(self.xs[0]))))
        yvalues = np.linspace(-6, 6, int(abs(6 * np.max(self.xs[0]))))
        x, y = np.meshgrid(xvalues, yvalues)
        z = fx(x, y)
        plt.contour(x, y, z)
        plt.plot(self.xs[:, 0], self.xs[:, 1])