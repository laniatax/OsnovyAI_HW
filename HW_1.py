import torch

# 1.1 Создание тензоров
tensor1 = torch.rand(3, 4)                     # Случайные числа от 0 до 1
tensor2 = torch.zeros(2, 3, 4)                 # Нули
tensor3 = torch.ones(5, 5)                     # Единицы
tensor4 = torch.arange(16).reshape(4, 4)       # От 0 до 15

# 1.2 Операции с тензорами
A = torch.rand(3, 4)
B = torch.rand(4, 3)
A_T = A.T                                      # Транспонирование
matmul_result = torch.matmul(A, B)             # Матричное умножение
elemwise_mult = A * B.T                        # Поэлементное умножение
sum_A = A.sum()                                # Сумма всех элементов

# 1.3 Индексация и срезы
tensor = torch.rand(5, 5, 5)
first_row = tensor[0, :, :]                    # Первая строка
last_column = tensor[:, :, -1]                 # Последний столбец
center = tensor[2:4, 2:4, 2:4]                 # Центр 2x2x2
even_indices = tensor[::2, ::2, ::2]           # Четные индексы

# 1.4 Работа с формами
flat_tensor = torch.arange(24)
reshape_2x12 = flat_tensor.reshape(2, 12)
reshape_3x8 = flat_tensor.reshape(3, 8)
reshape_4x6 = flat_tensor.reshape(4, 6)
reshape_2x3x4 = flat_tensor.reshape(2, 3, 4)
reshape_2x2x2x3 = flat_tensor.reshape(2, 2, 2, 3)


import torch

# 2.1 Градиенты простой функции
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = torch.tensor(3.0, requires_grad=True)
f = x**2 + y**2 + z**2 + 2 * x * y * z
f.backward()
# Градиенты: df/dx = 2x + 2yz, аналогично для y и z

# 2.2 MSE и градиенты
def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()

x = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([2.0, 4.0, 6.0])
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

y_pred = w * x + b
loss = mse_loss(y_pred, y_true)
loss.backward()
# Градиенты: w.grad, b.grad

# 2.3 Цепное правило
x = torch.tensor(1.0, requires_grad=True)
f = torch.sin(x**2 + 1)
f.backward()
# df/dx = cos(x^2 + 1) * 2x

# Проверка через grad
from torch.autograd import grad
grad_f = grad(f, x)[0]


import torch
import time

device_cpu = torch.device("cpu")
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3.1 Подготовка данных
shapes = [(64, 1024, 1024), (128, 512, 512), (256, 256, 256)]
tensors_cpu = [torch.rand(s) for s in shapes]
tensors_gpu = [t.to(device_gpu) for t in tensors_cpu]

# 3.2 Функция времени
def measure_time_cpu(func, *args):
    start = time.time()
    func(*args)
    return (time.time() - start) * 1000

def measure_time_gpu(func, *args):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    func(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)

# 3.3 Сравнение
def run_benchmark():
    operations = {
        "Матмул": lambda x: torch.matmul(x, x.transpose(-1, -2)),
        "Сложение": lambda x: x + x,
        "Умножение": lambda x: x * x,
        "Транспонирование": lambda x: x.transpose(-1, -2),
        "Сумма": lambda x: x.sum()
    }

    print(f"{'Операция':<18} | {'CPU (мс)':<10} | {'GPU (мс)':<10} | {'Ускорение'}")
    print("-" * 60)

    for op_name, op_func in operations.items():
        cpu_time = measure_time_cpu(op_func, tensors_cpu[0])
        if torch.cuda.is_available():
            gpu_time = measure_time_gpu(op_func, tensors_gpu[0])
            speedup = round(cpu_time / gpu_time, 2)
        else:
            gpu_time = float("nan")
            speedup = "N/A"
        print(f"{op_name:<18} | {cpu_time:<10.2f} | {gpu_time:<10.2f} | {speedup}")

run_benchmark()



"""
Анализ результатов:

1. Наибольшее ускорение на GPU достигается в матричном умножении и поэлементных операциях.
2. Некоторые операции, такие как транспонирование или сумма, могут быть медленнее на GPU при малых размерах данных из-за накладных расходов.
3. Ускорение растет с увеличением размеров матриц, т.к. GPU лучше справляется с массивными параллельными вычислениями.
4. Передача данных между CPU и GPU может занимать значительное время — по возможности избегать частых переключений.
"""
