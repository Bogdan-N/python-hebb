import numpy as np
from PIL import Image
import numpy as np

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        for p in patterns:
            p = np.reshape(p, (1, self.num_neurons))
            self.weights += np.dot(p.T, p)
            np.fill_diagonal(self.weights, 0)

    def recall(self, pattern):
        pattern = np.reshape(pattern, (1, self.num_neurons))
        for _ in range(5):  # Number of iterations for updating neurons
            s = np.dot(pattern, self.weights)
            pattern = np.sign(s)
        return pattern

def load_image(filename, threshold=128):
    img = Image.open(filename).convert('L')
    img_data = np.asarray(img)
    img_data_copy = img_data.copy()  # Create a deep copy
    img_data_copy[img_data_copy < threshold] = -1
    img_data_copy[img_data_copy >= threshold] = 1
    return img_data_copy.flatten()

# Завантаження зображень
letter_B = load_image('../letters/b.png')
letter_H = load_image('../letters/h.png')
letter_O = load_image('../letters/o.png')

patterns = [letter_H, letter_B, letter_O]

network = HopfieldNetwork(num_neurons=len(letter_H))
network.train(patterns)

input_pattern = load_image('../letters/b_test.png')
input_pattern = input_pattern.astype(np.float64)  # Змінюємо тип даних на float64
output = network.recall(input_pattern)
print("Output pattern:")
print(output)

recognized_letter = None
if np.array_equal(output, letter_B):
    recognized_letter = 'B'
elif np.array_equal(output, letter_H):
    recognized_letter = 'H'
elif np.array_equal(output, letter_O):
    recognized_letter = 'O'

if recognized_letter:
    print("Розпізнане зображення: літера", recognized_letter)
else:
    print("Розпізнане зображення: не вдалося розпізнати літеру")
