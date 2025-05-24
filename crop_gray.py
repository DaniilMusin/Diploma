# import os
# import numpy as np
# from PIL import Image

# def convert_to_grayscale(input_folder, output_folder):
#     # Проверяем существование входной папки
#     if not os.path.exists(input_folder):
#         print(f"Ошибка: Входная папка {input_folder} не существует")
#         return

#     # Создаем выходную папку, если она не существует
#     if not os.path.exists(output_folder):
#         try:
#             os.makedirs(output_folder)
#             print(f"Создана директория: {output_folder}")
#         except Exception as e:
#             print(f"Ошибка при создании директории {output_folder}: {str(e)}")
#             return

#     # Получаем список файлов
#     files = os.listdir(input_folder)
#     print(f"Найдено файлов во входной папке: {len(files)}")

#     # Перебираем все файлы в входной папке
#     processed_count = 0
#     for filename in files:
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
#             # Открываем изображение
#             img_path = os.path.join(input_folder, filename)
#             try:
#                 with Image.open(img_path) as img:
#                     # Конвертируем в RGB, если это не так
#                     if img.mode != 'RGB':
#                         img = img.convert('RGB')
                    
#                     # Преобразуем изображение в массив NumPy для применения формулы градаций серого
#                     img_array = np.array(img)
                    
#                     # Применяем формулу L = 0.299R + 0.587G + 0.114B для преобразования в градации серого
#                     r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
#                     grayscale = 0.299 * r + 0.587 * g + 0.114 * b
#                     grayscale = grayscale.astype(np.uint8)
                    
#                     # Создаем изображение в градациях серого
#                     gray_img = Image.fromarray(grayscale, mode='L')
                    
#                     # Сохраняем обработанное изображение
#                     output_path = os.path.join(output_folder, filename)
#                     gray_img.save(output_path)
                    
#                     processed_count += 1
#                     print(f"Обработано: {filename}")
#             except Exception as e:
#                 print(f"Ошибка при обработке {filename}: {str(e)}")
#         else:
#             print(f"Пропущен файл с неподдерживаемым расширением: {filename}")

#     print(f"Обработка завершена. Обработано {processed_count} изображений из {len(files)}.")
#     print(f"Изображения в градациях серого сохранены в {output_folder}")

# # Пути к папкам
# input_folder = r"D:\Diploma\Plates"
# output_folder = r"D:\Diploma\Plates\Processed_Grayscale_Images"

# # Вызов функции для преобразования в градации серого
# convert_to_grayscale(input_folder, output_folder)






import os
import numpy as np
from PIL import Image, ImageDraw

def crop_grayscale_images(input_folder, output_folder, crop_params_by_type):
    """
    Обрезает изображения в градациях серого по круговой области.
    
    Args:
        input_folder (str): Путь к папке с изображениями в градациях серого
        output_folder (str): Путь к папке для сохранения обрезанных изображений
        crop_params_by_type (dict): Словарь с параметрами обрезки для разных типов изображений
    """
    # Проверяем существование входной папки
    if not os.path.exists(input_folder):
        print(f"Ошибка: Входная папка {input_folder} не существует")
        return

    # Создаем выходную папку, если она не существует
    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
            print(f"Создана директория: {output_folder}")
        except Exception as e:
            print(f"Ошибка при создании директории {output_folder}: {str(e)}")
            return

    # Получаем список файлов
    files = os.listdir(input_folder)
    print(f"Найдено файлов во входной папке: {len(files)}")

    # Перебираем все файлы в входной папке
    processed_count = 0
    for filename in files:
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
                    # Убедимся, что изображение в градациях серого
                    if img.mode != 'L':
                        img = img.convert('L')
                    
                    # Создаем маску для круговой обрезки
                    mask = Image.new('L', img.size, 0)
                    draw = ImageDraw.Draw(mask)
                    draw.ellipse((center_x - radius, center_y - radius, 
                                center_x + radius, center_y + radius), fill=255)
                    
                    # Применяем маску
                    result = Image.new('L', img.size, 0)
                    result.paste(img, (0, 0), mask)
                    
                    # Обрезаем до размеров круга
                    result = result.crop((center_x - radius, center_y - radius, 
                                        center_x + radius, center_y + radius))
                    
                    # Сохраняем обработанное изображение
                    output_path = os.path.join(output_folder, filename)
                    result.save(output_path)
                    
                    processed_count += 1
                    print(f"Обработано: {filename}")
            except Exception as e:
                print(f"Ошибка при обработке {filename}: {str(e)}")
        else:
            print(f"Пропущен файл с неподдерживаемым расширением: {filename}")

    print(f"Обработка завершена. Обработано {processed_count} изображений из {len(files)}.")
    print(f"Обрезанные изображения сохранены в {output_folder}")

# Параметры обрезки для разных типов изображений
crop_params = {
    "Plate 1": (552, 538, 442),
    "Plate 2": (568, 455, 449),
    "Plate 8": (568, 455, 449),
    "default": (352, 250, 243)  # Параметры для остальных изображений
}

# Пути к папкам
input_folder = r"D:\Diploma\Plates\Processed_Grayscale_Images"
output_folder = r"D:\Diploma\Plates\Processed_Grayscale_Images_Cropped"

# Вызов функции для обрезки изображений
crop_grayscale_images(input_folder, output_folder, crop_params)
