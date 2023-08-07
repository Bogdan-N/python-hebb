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

    # Додаткова змінна для підрахунку кількості зображень для кожної літери
    num_images_per_letter = 2

    for i in range(outputs.shape[0]):
        # Повторення навчання на двох зображеннях для кожної літери
        for _ in range(num_images_per_letter):
            x = inputs[i]
            y = outputs[i]
            weights += np.outer(x, y)
    return weights


# Функція для розпізнавання букв за допомогою моделі Хебба
def predict_hebb_model(inputs, weights):
    
    activations = activation(np.dot(inputs, weights))
    return activations

# Завантаження та обробка зображень
def process_images(image_paths):
    images = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = img.flatten()
        img = np.where(img > 0, 1, -1)
        images.append(img)
    return np.array(images)

# Набір тренувальних даних
train_image_paths_b = ["./letters/b.png", "./letters/b_2.png"]
train_image_paths_d = ["./letters/d.png", "./letters/d_2.png"]
train_image_paths_h = ["./letters/h.png", "./letters/h_2.png"]
train_image_paths_o = ["./letters/o.png", "./letters/o_2.png"]

train_image_paths = [train_image_paths_b, train_image_paths_d, train_image_paths_h, train_image_paths_o]
# target_size = (5, 5)  # Цільовий розмір зображення

train_images = []
for paths in train_image_paths:
    images = process_images(paths)
    train_images.append(images)

inputs = np.vstack(train_images)
outputs = np.array([[1, -1, -1, -1],
                    [1, -1, -1, -1],
                    [-1, 1, -1, -1],
                    [-1, 1, -1, -1],
                    [-1, -1, 1, -1],
                    [-1, -1, 1, -1],
                    [-1, -1, -1, 1],
                    [-1, -1, -1, 1]])


# Навчання моделі Хебба
weights = train_hebb_model(inputs, outputs)

# Завантаження та обробка тестових зображень
test_image_paths = ["./letters/b_test.png", "./letters/d_test.png", "./letters/h_test.png", "./letters/o_test.png"]
test_images = process_images(test_image_paths)

# Тестування моделі Хебба
predictions = predict_hebb_model(test_images, weights)
print('predictions ==>>', predictions)

# Виведення результатів
print("Тестування розпізнавання:")

letters = ["б", "д", "г", "о"]
for i, prediction in enumerate(predictions):
    recognized_letter = letters[np.argmax(prediction)]
    expected_letter = letters[i]
    print(f"Тест {i + 1}: Розпізнано '{recognized_letter}', Очікувано '{expected_letter}'")
