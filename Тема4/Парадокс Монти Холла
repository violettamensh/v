import random
import matplotlib.pyplot as plt

N = 10000  # сколько раз играем
stay_wins = 0  # если не меняем
switch_wins = 0  # если меняем

# Играем N раз
for i in range(N):
    # где авто
    auto = random.randint(1, 3)

    # игрок выбирает
    choice = random.randint(1, 3)

    # если не меняет - победа когда угадал сразу
    if choice == auto:
        stay_wins += 1

    # если меняет - победа когда НЕ угадал сразу
    else:
        switch_wins += 1

# Результаты
print("=" * 40)
print("ПАРАДОКС МОНТИ ХОЛЛА")
print("=" * 40)
print(f"Всего игр: {N}")
print(f"Побед если НЕ менять: {stay_wins} ({stay_wins/N:.1%})")
print(f"Побед если менять: {switch_wins} ({switch_wins/N:.1%})")

plt.figure(figsize=(8, 5))

strategies = ['Не менять', 'Менять']
probabilities = [stay_wins/N, switch_wins/N]
colors = ['lightcoral', 'lightgreen']

bars = plt.bar(strategies, probabilities, color=colors, edgecolor='black')

for bar, prob in zip(bars, probabilities):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
            f'{prob:.1%}', ha='center', va='bottom', fontsize=12)

plt.axhline(y=1/3, color='red', linestyle='--', alpha=0.5, label='Теория: 33.3%')
plt.axhline(y=2/3, color='green', linestyle='--', alpha=0.5, label='Теория: 66.7%')

# Настройки графика
plt.title('Парадокс Монти Холла: менять или нет?', fontsize=14, pad=15)
plt.ylabel('Вероятность выигрыша', fontsize=12)
plt.ylim(0, 0.8)
plt.grid(axis='y', alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()
