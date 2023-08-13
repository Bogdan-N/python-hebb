import numpy as np
from PIL import Image

class HopfieldNetwork:
     def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

     def train(self, patterns):
        for p in patterns:
            p = np.reshape(p, (1, self.num_neurons))
            self.weights += np.dot(p.T, p)
            np.fill_diagonal(self.weights, 0)

     def recall(self, pattern, num_iterations=10, threshold=0.5):
        pattern = np.reshape(pattern, (1, self.num_neurons))
        for _ in range(num_iterations):
            s = np.dot(pattern, self.weights)
            pattern = np.sign(s)
            
            # Використання ваг віддаленості Хеммінга
            hamming_distances = np.sum(np.abs(pattern - self.weights), axis=1)
            pattern[0, hamming_distances > threshold] *= -1
        
        return pattern.astype(np.int8)

def load_image(filename, threshold=255):
    img = Image.open(filename).convert('L')
    img_data = np.asarray(img)
    print("image before convert ==> ")
    print(img_data)
    img_data_copy = img_data.copy()
    img_data_copy = np.where(img_data < threshold, -1, 1)
    print("image after convert ==> ")
    print(img_data_copy)
    return img_data_copy.flatten()

letter_B = load_image('../letters/b_2.png')
letter_H = load_image('../letters/h_2.png')
letter_O = load_image('../letters/o_2.png')

patterns = [letter_H, letter_B, letter_O]
network = HopfieldNetwork(num_neurons=len(letter_H))
network.train(patterns)

input_pattern = load_image('../letters/b_test.png')
input_pattern = input_pattern.astype(np.int8)
input_pattern = 2 * (input_pattern - 0.5)
input_pattern = np.reshape(input_pattern, (1, len(input_pattern)))
output = network.recall(input_pattern, num_iterations=10, threshold=0.5)
print("Output pattern:")
print(output)

print("Output letter_B:")
print(letter_B)

recognized_letter = None
if np.all(output == letter_B):
    recognized_letter = 'B'
elif np.all(output == letter_H):
    recognized_letter = 'H'
elif np.all(output == letter_O):
    recognized_letter = 'O'

if recognized_letter:
    print("Розпізнане зображення: літера", recognized_letter)
else:
    print("Розпізнане зображення: не вдалося розпізнати літеру")
