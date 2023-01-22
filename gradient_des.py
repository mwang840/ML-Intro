import numpy as np

x = np.array([2, 4, 8, 16, 32])
y = np.array([3, 9, 27, 81, 243])


def gradient_descent(x, y):
    m_curr = b_curr = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.0008

    for i in range(iterations):
        y_predict = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predict)])
        m_deriv = -(2 / n) * sum(x * (y - y_predict))
        b_deriv = -(2 / n) * sum((y - y_predict))
        m_curr = m_curr - learning_rate * m_deriv
        b_curr = b_curr - learning_rate * b_deriv
        print("m {}, b {}, cost {}, iteration {}".format(m_curr, b_curr, cost, i))


gradient_descent(x, y)
