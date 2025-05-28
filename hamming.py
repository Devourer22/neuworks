import numpy as np

class HammingNetwork:
    def __init__(self, patterns):
        self.patterns = np.array(patterns)
        self.num_patterns = len(patterns)
        self.input_size = len(patterns[0])

    def predict(self, input_vector):
        scores = [np.sum(np.logical_not(np.logical_xor(p, input_vector))) for p in self.patterns]
        best_match_idx = np.argmax(scores)
        return self.patterns[best_match_idx], best_match_idx, scores[best_match_idx]

def print_digit(vector):
    for i in range(5):
        row = vector[i * 4:(i + 1) * 4]
        print(''.join('1' if x else '.' for x in row))

# Эталоны 0–9
digits = {
    0: [0,1,1,0,
        1,0,0,1,
        1,0,0,1,
        1,0,0,1,
        0,1,1,0],

    1: [0,0,1,0,
        0,1,1,0,
        0,0,1,0,
        0,0,1,0,
        0,0,1,0],

    2: [0,1,1,0,
        1,0,0,1,
        0,0,1,0,
        0,1,0,0,
        1,1,1,1],

    3: [1,1,1,0,
        0,0,0,1,
        0,1,1,0,
        0,0,0,1,
        1,1,1,0],

    4: [0,0,1,0,
        0,1,1,0,
        1,0,1,0,
        1,1,1,1,
        0,0,1,0],

    5: [1,1,1,1,
        1,0,0,0,
        1,1,1,0,
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
        1,0,0,0],

    8: [0,1,1,0,
        1,0,0,1,
        0,1,1,0,
        1,0,0,1,
        0,1,1,0],

    9: [0,1,1,0,
        1,0,0,1,
        0,1,1,1,
        0,0,0,1,
        0,1,1,0]
}

# Подготовка сети
reference_patterns = list(digits.values())
hamming_net = HammingNetwork(reference_patterns)

# Искажённые образы
distorted_inputs = {
    "digit_1_missing_bottom": [0,0,1,0,
                               0,1,1,0,
                               0,0,1,0,
                               0,0,1,0,
                               0,0,0,0],

    "digit_2_noise_top":      [0,1,1,1,
                               1,0,0,1,
                               0,0,1,0,
                               0,1,0,0,
                               1,1,1,1],

    "digit_3_shifted":        [1,1,1,0,
                               0,0,0,0,
                               0,1,1,0,
                               0,0,0,1,
                               1,1,1,0],

    "digit_8_noise":          [1,1,1,0,
                               1,0,0,1,
                               0,1,1,0,
                               1,0,0,1,
                               1,1,1,0],

    "digit_4_corrupted":      [0,0,0,0,
                               1,1,1,0,
                               1,0,1,0,
                               1,1,1,1,
                               0,0,1,0],

    "digit_6_half_missing":   [0,0,0,0,
                               0,0,0,0,
                               1,1,1,0,
                               1,0,0,1,
                               0,1,1,0],

    "digit_0_broken_edges":   [0,1,0,0,
                               1,0,0,1,
                               1,0,0,1,
                               1,0,0,1,
                               0,1,1,0],

    "digit_5_exact":          digits[5],

    "digit_9_heavily_corrupted": [1,0,1,0,
                                  1,1,0,1,
                                  0,1,0,1,
                                  0,0,0,0,
                                  1,1,0,1],
}

# Проверка
for name, distorted in distorted_inputs.items():
    print(f"\n{name}:")
    print("Входной образ:")
    print_digit(distorted)

    predicted, index, score = hamming_net.predict(distorted)
    print("\nВосстановленный образ:")
    print_digit(predicted)
    print(f"\nСовпадает с цифрой: {index} (совпавших битов: {score})")
    correct = np.array_equal(predicted, distorted)
    print("Совпадает с входом:", correct)
