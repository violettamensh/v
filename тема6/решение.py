import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous
from scipy import integrate

# СОЗДАНИЕ СОБСТВЕННОГО РАСПРЕДЕЛЕНИЯ

# f(x) = 3/2 * (1 - x^2), x ∈ [0, 1]

class CustomDistribution(rv_continuous):
    def __init__(self, a, b):
        super().__init__(a=a, b=b, name='custom_dist')

    def _pdf(self, x):
        """Плотность распределения"""
        result = np.zeros_like(x)
        mask = (x >= self.a) & (x <= self.b)
        result[mask] = (3/2) * (1 - x[mask]**2)
        return result

    def _cdf(self, x):
        """Функция распределения"""
        result = np.zeros_like(x)
        result[x < self.a] = 0
        mask = (x >= self.a) & (x <= self.b)
        result[mask] = (3/2) * (x[mask] - x[mask]**3/3)
        result[x > self.b] = 1
        return result

    def _ppf(self, q):
        """Обратная функция распределения (квантильная функция)"""
        # Для уравнения (3/2)*(x - x³/3) = q используем численное решение
        result = np.zeros_like(q)
        for i, q_val in enumerate(q):
            # Начальное приближение
            x0 = 0.5
            # Несколько итераций методом Ньютона
            for _ in range(10):
                f = (3/2)*(x0 - x0**3/3) - q_val
                f_prime = (3/2)*(1 - x0**2)
                if abs(f_prime) < 1e-10:
                    break
                x0 = x0 - f/f_prime
                x0 = max(self.a, min(self.b, x0))  # Ограничиваем границами
            result[i] = x0
        return result

# СОЗДАНИЕ ЭКЗЕМПЛЯРА РАСПРЕДЕЛЕНИЯ

a, b = 0.0, 1.0
custom_dist = CustomDistribution(a, b)

# Генерация выборки
np.random.seed(42)
sample = custom_dist.rvs(size=10000)

print(f"Размер выборки: {len(sample)}")
print(f"Первые 10 значений: {sample[:10].round(4)}")

x_test = np.linspace(a - 0.5, b + 0.5, 100)
pdf_values = custom_dist.pdf(x_test)
cdf_values = custom_dist.cdf(x_test)

print("Проверка нормировки плотности:")
integral, error = integrate.quad(custom_dist.pdf, a, b)
print(f"∫f(x)dx на [{a}, {b}] = {integral:.6f} (ошибка: {error:.2e})")

print("\nПроверка в граничных точках:")
print(f"F({a}) = {custom_dist.cdf(a):.6f}")
print(f"F({b}) = {custom_dist.cdf(b):.6f}")

# ВЕРОЯТНОСТЬ ПОПАДАНИЯ В ИНТЕРВАЛ

# Рассчитаем P(0.2 < X < 0.8)
x1, x2 = 0.2, 0.8
prob_interval = custom_dist.cdf(x2) - custom_dist.cdf(x1)
prob_empirical = np.mean((sample > x1) & (sample < x2))

print(f"\nP({x1} < X < {x2}) = {prob_interval:.6f} (теоретическая)")
print(f"P({x1} < X < {x2}) = {prob_empirical:.6f} (эмпирическая)")

# ЧИСЛОВЫЕ ХАРАКТЕРИСТИКИ

print("\n" + "ЧИСЛОВЫЕ ХАРАКТЕРИСТИКИ".center(50, '*'))

print(f"Математическое ожидание: {custom_dist.mean():.6f}")
print(f"Дисперсия: {custom_dist.var():.6f}")
print(f"Среднее квадратическое отклонение: {custom_dist.std():.6f}")
print(f"Медиана: {custom_dist.median():.6f}")

# КВАНТИЛИ И ПРОЦЕНТНЫЕ ТОЧКИ

q = 0.3  # уровень квантиля
p = 80   # процент для p%-ной точки

quantile_q = custom_dist.ppf([q])[0]
p_point = custom_dist.ppf([p/100])[0]

print(f"\n{q:.0%}-квантиль: {quantile_q:.6f}")
print(f"{p}%-ная точка: {p_point:.6f}")

# КОЭФФИЦИЕНТ АСИММЕТРИИ И ЭКСЦЕСС

print("\n" + "КОЭФФИЦИЕНТЫ".center(50, '*'))

mean, variance, skewness, kurtosis = custom_dist.stats(moments='mvsk')
print(f"Коэффициент асимметрии: {skewness:.6f}")
print(f"Эксцесс: {kurtosis:.6f}")

print("\nСтатистики (среднее, дисперсия, асимметрия, эксцесс):")
print(f"  Среднее: {mean:.6f}")
print(f"  Дисперсия: {variance:.6f}")
print(f"  Асимметрия: {skewness:.6f}")
print(f"  Эксцесс: {kurtosis:.6f}")

# ПОСТРОЕНИЕ ГРАФИКОВ

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
x_plot = np.linspace(a - 0.2, b + 0.2, 100)

# 1. Функция распределения
ax1.hist(sample, bins=50, density=True, cumulative=True,
         alpha=0.7, label='Эмпирическая CDF')
ax1.plot(x_plot, custom_dist.cdf(x_plot), 'r-', linewidth=2,
         label='Теоретическая CDF')
ax1.set_xlabel('x')
ax1.set_ylabel('F(x)')
ax1.set_title('Функция распределения')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Гистограмма и теоретическая плотность
ax2.hist(sample, bins=50, density=True, alpha=0.7, label='Выборка')
ax2.plot(x_plot, custom_dist.pdf(x_plot), 'r-', linewidth=2,
         label='Теоретическая PDF')
ax2.set_xlabel('x')
ax2.set_ylabel('Плотность вероятности')
ax2.set_title('Гистограмма и теоретическая плотность')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
