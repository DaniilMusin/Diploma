import os
import numpy as np
from PIL import Image, ImageDraw

def crop_and_convert_to_grayscale(input_folder, output_folder, crop_params_by_type):
    """
    Обрезает изображения по круговой области и преобразует их в градации серого.
    
    Args:
        input_folder (str): Путь к папке с исходными изображениями
        output_folder (str): Путь к папке для сохранения обработанных изображений
        crop_params_by_type (dict): Словарь с параметрами обрезки для разных типов изображений
    """
    # Создаем выходную папку, если она не существует
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Перебираем все файлы в входной папке
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Определяем параметры обрезки в зависимости от имени файла
            if "Plate 1" in filename:
                center_x, center_y, radius = crop_params_by_type["Plate 1"]
            elif "Plate 2" in filename:
                center_x, center_y, radius = crop_params_by_type["Plate 2"]
            elif "Plate 8" in filename:
                center_x, center_y, radius = crop_params_by_type["Plate 8"]
            else:
                center_x, center_y, radius = crop_params_by_type["default"]
            
            # Открываем изображение
            img_path = os.path.join(input_folder, filename)
            try:
                with Image.open(img_path) as img:
                    # Конвертируем в RGB, если это не так (для обработки RGBA, индексированных и других форматов)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Преобразуем изображение в массив NumPy для применения формулы градаций серого
                    img_array = np.array(img)
                    
                    # Применяем формулу L = 0.299R + 0.587G + 0.114B для преобразования в градации серого
                    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
                    grayscale = 0.299 * r + 0.587 * g + 0.114 * b
                    grayscale = grayscale.astype(np.uint8)
                    
                    # Создаем изображение в градациях серого
                    gray_img = Image.fromarray(grayscale, mode='L')
                    
                    # Создаем маску для круговой обрезки
                    mask = Image.new('L', gray_img.size, 0)
                    draw = ImageDraw.Draw(mask)
                    draw.ellipse((center_x - radius, center_y - radius, 
                                center_x + radius, center_y + radius), fill=255)

                    # Применяем маску
                    output = Image.new('L', gray_img.size, 0)
                    output.paste(gray_img, (0, 0), mask)

                    # Обрезаем изображение до размеров круга
                    output = output.crop((center_x - radius, center_y - radius, 
                                        center_x + radius, center_y + radius))

                    # Сохраняем обработанное изображение
                    output_path = os.path.join(output_folder, f"processed_{filename.split('.')[0]}.png")
                    output.save(output_path)
                    
                    print(f"Обработано: {filename}")
            except Exception as e:
                print(f"Ошибка при обработке {filename}: {str(e)}")

    print(f"Обработка завершена. Изображения сохранены в {output_folder}")

# Параметры обрезки для разных типов изображений
crop_params = {
    "Plate 1": (552, 538, 442),
    "Plate 2": (568, 455, 449),
    "Plate 8": (568, 455, 449),
    "default": (352, 250, 243)  # Параметры для остальных изображений
}

# Пути к папкам
input_folder = "C:\\Diploma\\gold_set"
output_folder = os.path.join(input_folder, "Processed_Grayscale_Images")

# Вызов функции для обрезки изображений и преобразования в градации серого
crop_and_convert_to_grayscale(input_folder, output_folder, crop_params)
