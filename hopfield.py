import numpy as np

def sign(x):
    return np.where(x >= 0, 1, -1)

def bin_to_bipolar(vector):
    return np.array([1 if x == 1 else -1 for x in vector])

def bipolar_to_bin(vector):
    return np.where(vector > 0, 1, 0)

def print_digit(vector):
    for i in range(5):
        row = vector[i*4:(i+1)*4]
        print(''.join('1' if x else '.' for x in row))

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for p in patterns:
            p = bin_to_bipolar(p)
            self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0)

    def predict(self, input_vector, steps=10):
        s = bin_to_bipolar(input_vector.copy())
        for _ in range(steps):
            s = sign(np.dot(self.weights, s))
        return bipolar_to_bin(s)

digits = {
    1: [0,0,1,0,
        0,1,1,0,
        0,0,1,0,
        0,0,1,0,
        0,0,1,0],

    3: [1,1,1,0,
        0,0,0,1,
        0,1,1,0,
        0,0,0,1,
        1,1,1,0],

    6: [0,1,1,0,
        1,0,0,0,
        1,1,1,0,
        1,0,0,1,
        0,1,1,0],

    7: [1,1,1,1,
        0,0,0,1,
        0,0,1,0,
        0,1,0,0,
        1,0,0,0]
}

network = HopfieldNetwork(size=20)
patterns = list(digits.values())
network.train(patterns)

distorted_digits = {
    "digit_1_missing_bottom": [0,0,1,0,
                               0,1,1,0,
                               0,0,1,0,
                               0,0,1,0,
                               0,0,0,0],  # лёгкое искажение — восстанавливается

    "digit_3_noise_top":      [1,1,1,1,
                               0,0,0,1,
                               0,1,1,0,
                               0,0,0,1,
                               1,1,1,0],  # лёгкий шум — восстанавливается

    "digit_6_easy_noise":     [0,1,1,0,
                               1,0,0,0,
                               1,1,1,0,
                               1,0,0,1,
                               0,1,0,0],  # восстанавливается

    # Ниже искажения сильные — сеть не справится
    "digit_6_scrambled":      [0,0,0,0,
                               0,0,0,0,
                               1,1,0,1,
                               0,0,1,1,
                               1,0,0,0],  # вырезана нижняя часть, искажен верх

    "digit_1_removed_top":    [0,0,0,0,
                               0,0,0,0,
                               0,0,1,0,
                               0,0,1,0,
                               0,0,1,0],  # удалена верхушка — не узнается

    "digit_3_flattened":      [0,0,0,0,
                               0,1,1,0,
                               0,1,1,0,
                               0,1,1,0,
                               0,0,0,0],  # «плоский» шум вместо характерного вида

    "digit_7_flipped_tail":   [1,1,1,1,
                               0,0,0,1,
                               0,0,1,0,
                               1,0,0,0,
                               0,1,0,0]  # перепутан хвост, сетка не узнает
}


def recognize_digit(output_vector, reference_digits):
    for digit, ref in reference_digits.items():
        if np.array_equal(output_vector, ref):
            return f"{digit} (УСПЕХ)"
    return "Не распознано (ОШИБКА)"

for name, distorted in distorted_digits.items():
    print(f"\n{name}:")
    print("Входной образ:")
    print_digit(distorted)

    restored = network.predict(distorted)
    print("\nВосстановленный образ:")
    print_digit(restored)

    result = recognize_digit(restored, digits)
    print("\nРаспознано как:", result)

