import cv2
import numpy as np

# Функція активації для моделі Хебба
def activation(x):
    return np.where(x >= 0, 1, -1)

# Функція для навчання моделі Хебба
def train_hebb_model(inputs, outputs):
    num_inputs = inputs.shape[1]
    num_outputs = outputs.shape[1]
    weights = np.zeros((num_inputs, num_outputs))

    for _ in range(2):  # Тренування на 2 зображеннях для однієї букви
        for i in range(outputs.shape[0]):
            x = inputs[i]
            y = outputs[i]
            weights += np.outer(x, y)
          
    return weights

# Функція для розпізнавання букв за допомогою моделі Хебба
def predict_hebb_model(inputs, weights):
    activations = activation(np.dot(inputs, weights))
    return activations

# Завантаження та обробка зображення
def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = img.flatten()
    img = np.where(img > 0, 1, -1)
    return img

# Набір тренувальних даних (виберіть одну букву)
train_image_paths = ["./letters/d.png", "./letters/d_2.png"]
# target_size = (5, 5)  # Цільовий розмір зображення

# Завантаження та обробка першого зображення
train_image_1 = process_image(train_image_paths[0])
# Завантаження та обробка другого зображення
train_image_2 = process_image(train_image_paths[1])

inputs = np.vstack([train_image_1, train_image_2])
outputs = np.array([[1, -1, -1, -1]])

# Навчання моделі Хебба
weights = train_hebb_model(inputs, outputs)

# Завантаження та обробка тестових зображень (виберіть одну букву для тестування)
test_image_paths = ["./letters/o_test.png"]
test_image = process_image(test_image_paths[0])

# Тестування моделі Хебба
predictions = predict_hebb_model(test_image, weights)
print('****found****', predictions)
indices = np.where(np.all(outputs == predictions, axis=1))
if len(indices):
    print('****found****')
else:
    print('****NOT found****')
found_index = indices[0][0]
print('indices ==>> ', found_index)

letters = ["б", "д", "г", "о"]
print('indices ==>> ', letters[found_index])

# letters = ["б", "д", "г", "о"]

# for i, prediction in enumerate(predictions):
#     recognized_letter = letters[np.argmax(prediction)]
#     expected_letter = letters[i]
#     print(f"Тест {i + 1}: Розпізнано '{recognized_letter}', Очікувано '{expected_letter}'")
