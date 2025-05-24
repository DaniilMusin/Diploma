import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import json
import datetime

class CircularCropperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Круговой обрезчик изображений")
        self.root.geometry("1200x700")
        
        self.images = []
        self.current_image_index = 0
        self.circle_center = None
        self.circle_radius = 0
        self.is_drawing = False
        
        # Создаем фреймы
        self.top_frame = tk.Frame(root)
        self.top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.bottom_frame = tk.Frame(root)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        # Кнопки верхнего фрейма
        self.load_btn = tk.Button(self.top_frame, text="Загрузить изображения", command=self.load_images)
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        self.prev_btn = tk.Button(self.top_frame, text="Предыдущее", command=self.prev_image, state=tk.DISABLED)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        self.next_btn = tk.Button(self.top_frame, text="Следующее", command=self.next_image, state=tk.DISABLED)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        self.image_label = tk.Label(self.top_frame, text="Изображение: 0/0")
        self.image_label.pack(side=tk.LEFT, padx=20)
        
        # Холст для отображения изображения
        self.canvas = tk.Canvas(self.canvas_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Привязываем события мыши
        self.canvas.bind("<ButtonPress-1>", self.start_circle)
        self.canvas.bind("<B1-Motion>", self.draw_circle)
        self.canvas.bind("<ButtonRelease-1>", self.end_circle)
        
        # Кнопки нижнего фрейма
        self.crop_btn = tk.Button(self.bottom_frame, text="Обрезать текущее", command=self.crop_current, state=tk.DISABLED)
        self.crop_btn.pack(side=tk.LEFT, padx=5)
        
        self.crop_all_btn = tk.Button(self.bottom_frame, text="Обрезать все", command=self.crop_all, state=tk.DISABLED)
        self.crop_all_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_btn = tk.Button(self.bottom_frame, text="Сбросить выделение", command=self.reset_selection, state=tk.DISABLED)
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = tk.Button(self.bottom_frame, text="Сохранить изображения", command=self.save_images, state=tk.DISABLED)
        self.save_btn.pack(side=tk.RIGHT, padx=5)
        
        self.save_params_btn = tk.Button(self.bottom_frame, text="Сохранить параметры", command=self.save_crop_parameters, state=tk.DISABLED)
        self.save_params_btn.pack(side=tk.RIGHT, padx=5)
        
        self.find_best_crop_btn = tk.Button(self.bottom_frame, text="Найти оптимальную обрезку", command=self.find_best_crop, state=tk.DISABLED)
        self.find_best_crop_btn.pack(side=tk.RIGHT, padx=5)
        
        # Данные для хранения
        self.original_images = []
        self.image_paths = []
        self.cropped_images = []
        self.circle_data = None
        self.canvas_image = None
        self.canvas_image_id = None
        self.circle_id = None
        
        # Словарь для хранения параметров обрезки для каждого изображения
        self.crop_parameters = {}
        
        # Оптимальные параметры обрезки (наибольший круг)
        self.best_crop_params = None
    
    def load_images(self):
        filepaths = filedialog.askopenfilenames(
            initialdir="C:/Diploma/gold_set",
            title="Выберите изображения",
            filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*"))
        )
        
        if not filepaths:
            return
        
        self.original_images = []
        self.image_paths = []
        self.cropped_images = []
        self.crop_parameters = {}
        
        for filepath in filepaths:
            try:
                img = Image.open(filepath).convert("RGBA")
                self.original_images.append(img)
                self.image_paths.append(filepath)
                self.cropped_images.append(None)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить {filepath}: {str(e)}")
        
        if self.original_images:
            self.current_image_index = 0
            self.update_image_display()
            self.update_navigation_buttons()
            self.reset_btn.config(state=tk.NORMAL)
            self.find_best_crop_btn.config(state=tk.NORMAL)
    
    def update_image_display(self):
        if not self.original_images:
            return
        
        img = self.original_images[self.current_image_index]
        
        # Масштабируем изображение для отображения на холсте
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Если холст еще не отрисован, используем значения по умолчанию
            canvas_width = 800
            canvas_height = 500
        
        img_width, img_height = img.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        self.display_img = img.resize((new_width, new_height), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(self.display_img)
        
        # Очищаем холст и отображаем изображение
        self.canvas.delete("all")
        self.canvas_image_id = self.canvas.create_image(
            canvas_width // 2, canvas_height // 2, 
            image=self.tk_img, anchor=tk.CENTER
        )
        
        # Обновляем метку с индексом изображения
        self.image_label.config(
            text=f"Изображение: {self.current_image_index + 1}/{len(self.original_images)} - {os.path.basename(self.image_paths[self.current_image_index])}"
        )
        
        # Проверяем, есть ли параметры обрезки для текущего изображения
        current_path = self.image_paths[self.current_image_index]
        if current_path in self.crop_parameters:
            self.circle_data = self.crop_parameters[current_path]
            center_x, center_y, radius = self.circle_data
            
            # Масштабируем координаты круга
            scale_x = new_width / img_width
            scale_y = new_height / img_height
            
            scaled_center_x = center_x * scale_x
            scaled_center_y = center_y * scale_y
            scaled_radius = radius * scale_x  # Используем scale_x для сохранения пропорций
            
            # Вычисляем координаты для холста
            canvas_center_x = canvas_width // 2 - new_width // 2 + scaled_center_x
            canvas_center_y = canvas_height // 2 - new_height // 2 + scaled_center_y
            
            # Рисуем круг
            self.circle_id = self.canvas.create_oval(
                canvas_center_x - scaled_radius,
                canvas_center_y - scaled_radius,
                canvas_center_x + scaled_radius,
                canvas_center_y + scaled_radius,
                outline="red", width=2
            )
        else:
            self.circle_data = None
            self.circle_id = None
    
    def update_navigation_buttons(self):
        if not self.original_images:
            self.prev_btn.config(state=tk.DISABLED)
            self.next_btn.config(state=tk.DISABLED)
            self.crop_btn.config(state=tk.DISABLED)
            self.crop_all_btn.config(state=tk.DISABLED)
            self.save_btn.config(state=tk.DISABLED)
            self.save_params_btn.config(state=tk.DISABLED)
            return
        
        if self.current_image_index > 0:
            self.prev_btn.config(state=tk.NORMAL)
        else:
            self.prev_btn.config(state=tk.DISABLED)
        
        if self.current_image_index < len(self.original_images) - 1:
            self.next_btn.config(state=tk.NORMAL)
        else:
            self.next_btn.config(state=tk.DISABLED)
        
        self.crop_btn.config(state=tk.NORMAL)
        self.crop_all_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)
        self.save_params_btn.config(state=tk.NORMAL)
    
    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.update_image_display()
            self.update_navigation_buttons()
    
    def next_image(self):
        if self.current_image_index < len(self.original_images) - 1:
            self.current_image_index += 1
            self.update_image_display()
            self.update_navigation_buttons()
    
    def start_circle(self, event):
        if not self.original_images:
            return
        
        # Получаем координаты холста
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Получаем размеры отображаемого изображения
        img_width, img_height = self.display_img.size
        
        # Вычисляем смещение изображения на холсте
        offset_x = (canvas_width - img_width) // 2
        offset_y = (canvas_height - img_height) // 2
        
        # Проверяем, что клик был внутри изображения
        if (offset_x <= event.x <= offset_x + img_width and
            offset_y <= event.y <= offset_y + img_height):
            
            # Сохраняем начальную точку (центр круга)
            self.circle_center = (event.x - offset_x, event.y - offset_y)
            self.is_drawing = True
            
            # Удаляем предыдущий круг, если он есть
            if self.circle_id:
                self.canvas.delete(self.circle_id)
            
            # Рисуем начальный круг
            self.circle_id = self.canvas.create_oval(
                event.x, event.y, event.x + 1, event.y + 1,
                outline="red", width=2
            )
    
    def draw_circle(self, event):
        if not self.is_drawing or not self.circle_center:
            return
        
        # Получаем координаты холста
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Получаем размеры отображаемого изображения
        img_width, img_height = self.display_img.size
        
        # Вычисляем смещение изображения на холсте
        offset_x = (canvas_width - img_width) // 2
        offset_y = (canvas_height - img_height) // 2
        
        # Вычисляем радиус круга
        dx = event.x - (self.circle_center[0] + offset_x)
        dy = event.y - (self.circle_center[1] + offset_y)
        self.circle_radius = int(np.sqrt(dx**2 + dy**2))
        
        # Обновляем круг
        center_x = self.circle_center[0] + offset_x
        center_y = self.circle_center[1] + offset_y
        
        self.canvas.delete(self.circle_id)
        self.circle_id = self.canvas.create_oval(
            center_x - self.circle_radius,
            center_y - self.circle_radius,
            center_x + self.circle_radius,
            center_y + self.circle_radius,
            outline="red", width=2
        )
    
    def end_circle(self, event):
        if not self.is_drawing:
            return
        
        self.is_drawing = False
        
        # Масштабируем координаты обратно к оригинальному изображению
        if self.circle_center and self.circle_radius > 0:
            img = self.original_images[self.current_image_index]
            img_width, img_height = img.size
            display_width, display_height = self.display_img.size
            
            scale_x = img_width / display_width
            scale_y = img_height / display_height
            
            original_center_x = int(self.circle_center[0] * scale_x)
            original_center_y = int(self.circle_center[1] * scale_y)
            original_radius = int(self.circle_radius * scale_x)  # Используем scale_x для сохранения пропорций
            
            self.circle_data = (original_center_x, original_center_y, original_radius)
            
            # Сохраняем параметры обрезки для текущего изображения
            current_path = self.image_paths[self.current_image_index]
            self.crop_parameters[current_path] = self.circle_data
    
    def reset_selection(self):
        if self.circle_id:
            self.canvas.delete(self.circle_id)
            self.circle_id = None
        
        self.circle_center = None
        self.circle_radius = 0
        self.circle_data = None
        
        # Удаляем параметры обрезки для текущего изображения
        current_path = self.image_paths[self.current_image_index]
        if current_path in self.crop_parameters:
            del self.crop_parameters[current_path]
    
    def crop_current(self):
        if not self.original_images:
            messagebox.showwarning("Предупреждение", "Нет загруженных изображений")
            return
        
        current_path = self.image_paths[self.current_image_index]
        if current_path not in self.crop_parameters:
            messagebox.showwarning("Предупреждение", "Сначала нарисуйте круг для обрезки")
            return
        
        img = self.original_images[self.current_image_index]
        center_x, center_y, radius = self.crop_parameters[current_path]
        
        # Создаем маску для круговой обрезки
        mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((center_x - radius, center_y - radius, 
                      center_x + radius, center_y + radius), fill=255)
        
        # Создаем прозрачный фон
        background = Image.new("RGBA", img.size, (0, 0, 0, 0))
        
        # Применяем маску
        cropped_img = Image.composite(img, background, mask)
        
        # Сохраняем обрезанное изображение
        self.cropped_images[self.current_image_index] = cropped_img
        
        # Показываем результат
        messagebox.showinfo("Успех", "Изображение успешно обрезано")
    
    def crop_all(self):
        if not self.original_images:
            messagebox.showwarning("Предупреждение", "Нет загруженных изображений")
            return
        
        if not self.crop_parameters and not self.best_crop_params:
            messagebox.showwarning("Предупреждение", "Сначала нарисуйте круг для обрезки или найдите оптимальную обрезку")
            return
        
        # Если есть оптимальные параметры обрезки, используем их
        if self.best_crop_params:
            center_x, center_y, radius = self.best_crop_params
            
            for i, img in enumerate(self.original_images):
                # Создаем маску для круговой обрезки
                mask = Image.new("L", img.size, 0)
                draw = ImageDraw.Draw(mask)
                draw.ellipse((center_x - radius, center_y - radius, 
                              center_x + radius, center_y + radius), fill=255)
                
                # Создаем
# Создаем прозрачный фон
                background = Image.new("RGBA", img.size, (0, 0, 0, 0))
                
                # Применяем маску
                cropped_img = Image.composite(img, background, mask)
                
                # Сохраняем обрезанное изображение
                self.cropped_images[i] = cropped_img
            
            messagebox.showinfo("Успех", "Все изображения успешно обрезаны с использованием оптимальных параметров")
        else:
            # Проверяем, что у нас есть параметры обрезки хотя бы для одного изображения
            if not self.crop_parameters:
                messagebox.showwarning("Предупреждение", "Сначала нарисуйте круг для обрезки хотя бы для одного изображения")
                return
            
            # Используем параметры обрезки для каждого изображения
            for i, img in enumerate(self.original_images):
                path = self.image_paths[i]
                
                # Если для этого изображения нет параметров обрезки, используем параметры текущего изображения
                if path not in self.crop_parameters:
                    current_path = self.image_paths[self.current_image_index]
                    if current_path in self.crop_parameters:
                        self.crop_parameters[path] = self.crop_parameters[current_path]
                    else:
                        # Если нет параметров для текущего изображения, используем первые доступные параметры
                        first_path = next(iter(self.crop_parameters))
                        self.crop_parameters[path] = self.crop_parameters[first_path]
                
                center_x, center_y, radius = self.crop_parameters[path]
                
                # Создаем маску для круговой обрезки
                mask = Image.new("L", img.size, 0)
                draw = ImageDraw.Draw(mask)
                draw.ellipse((center_x - radius, center_y - radius, 
                              center_x + radius, center_y + radius), fill=255)
                
                # Создаем прозрачный фон
                background = Image.new("RGBA", img.size, (0, 0, 0, 0))
                
                # Применяем маску
                cropped_img = Image.composite(img, background, mask)
                
                # Сохраняем обрезанное изображение
                self.cropped_images[i] = cropped_img
            
            messagebox.showinfo("Успех", "Все изображения успешно обрезаны")
    
    def find_best_crop(self):
        """Находит оптимальные параметры обрезки (наименьший круг, который включает все области)"""
        if not self.crop_parameters:
            messagebox.showwarning("Предупреждение", "Сначала нарисуйте круг для обрезки хотя бы для одного изображения")
            return
        
        # Находим наименьший радиус из всех параметров обрезки
        min_radius = float('inf')
        best_center_x = 0
        best_center_y = 0
        best_radius = 0
        
        for path, (center_x, center_y, radius) in self.crop_parameters.items():
            if radius < min_radius:
                min_radius = radius
                best_center_x = center_x
                best_center_y = center_y
                best_radius = radius
        
        self.best_crop_params = (best_center_x, best_center_y, best_radius)
        
        # Обновляем отображение с новыми параметрами
        self.circle_data = self.best_crop_params
        self.update_image_display()
        
        messagebox.showinfo("Успех", f"Найдены оптимальные параметры обрезки: центр ({best_center_x}, {best_center_y}), радиус {best_radius}")
    
    def save_images(self):
        if not any(self.cropped_images):
            messagebox.showwarning("Предупреждение", "Нет обрезанных изображений для сохранения")
            return
        
        save_dir = filedialog.askdirectory(title="Выберите папку для сохранения")
        
        if not save_dir:
            return
        
        for i, img in enumerate(self.cropped_images):
            if img:
                # Получаем оригинальное имя файла
                original_filename = os.path.basename(self.image_paths[i])
                name, ext = os.path.splitext(original_filename)
                
                # Создаем имя файла
                filename = f"{name}_cropped.png"
                filepath = os.path.join(save_dir, filename)
                
                # Сохраняем изображение
                img.save(filepath)
        
        messagebox.showinfo("Успех", f"Изображения сохранены в {save_dir}")
    
    def save_crop_parameters(self):
        """Сохраняет параметры обрезки в JSON-файл"""
        if not self.crop_parameters and not self.best_crop_params:
            messagebox.showwarning("Предупреждение", "Нет параметров обрезки для сохранения")
            return
        
        save_path = filedialog.asksaveasfilename(
            title="Сохранить параметры обрезки",
            defaultextension=".json",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
        )
        
        if not save_path:
            return
        
        # Подготавливаем данные для сохранения
        data = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "individual_parameters": {},
            "best_parameters": None
        }
        
        # Сохраняем индивидуальные параметры для каждого изображения
        for path, params in self.crop_parameters.items():
            # Используем только имя файла вместо полного пути
            filename = os.path.basename(path)
            data["individual_parameters"][filename] = {
                "center_x": params,
                "center_y": params,
                "radius": params
            }
        
        # Сохраняем оптимальные параметры, если они есть
        if self.best_crop_params:
            data["best_parameters"] = {
                "center_x": self.best_crop_params,
                "center_y": self.best_crop_params,
                "radius": self.best_crop_params
            }
        
        # Записываем данные в файл
        try:
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=4)
            messagebox.showinfo("Успех", f"Параметры обрезки сохранены в {save_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить параметры: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CircularCropperApp(root)
    root.mainloop()

