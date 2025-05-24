import sys
import serial
import serial.tools.list_ports
import threading
import time
import datetime
import os

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QLineEdit, QPushButton, QGridLayout, QTextEdit
from PyQt5.QtCore import Qt, QTimer, QMetaObject, Q_ARG

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ArduinoGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.serial_conn = None  # Initialize serial connection as None
        self.initUI()

        # Автоматически найти порт Arduino
        self.serial_conn = self.find_arduino_port()
        if not self.serial_conn:
            self.log("Arduino не найдена. Пожалуйста, проверьте подключение.")
        else:
            self.log("Arduino подключена и готова к работе.")
            # Запуск фонового потока для чтения данных с Arduino
            self.read_thread = threading.Thread(target=self.read_from_arduino)
            self.read_thread.daemon = True
            self.read_thread.start()

        # Инициализация данных для графика
        self.start_time = time.time()
        self.xdata = []
        self.ydata = []

        # Таймер для обновления графика
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1000)  # Обновление каждую секунду

        # Настройка файла лога
        self.log_file = self.setup_log_file()

    def setup_log_file(self):
        # Перемещение в каталог, где находится скрипт
        script_dir = os.path.dirname(os.path.realpath(__file__))
        os.chdir(script_dir)
        self.log(f"Текущий рабочий каталог изменен на: {os.getcwd()}")

        # Получение текущей даты и времени
        current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        # Создание имени файла
        file_name = f"temperature_log_{current_time}.txt"
        self.log(f"Создание файла лога: {file_name}")

        try:
            log_file = open(file_name, "w", encoding="utf-8")
            self.log("Файл лога успешно создан.")
            # Запись заголовков колонок
            log_file.write("Время (с)\tТемпература (°C)\tМощность (ШИМ)\n")
            return log_file
        except Exception as e:
            self.log(f"Ошибка при создании файла лога: {e}")
            return None

    def initUI(self):
        self.setWindowTitle('Управление температурой Arduino')
        self.setGeometry(100, 100, 1200, 600)

        # Основной виджет и макет
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QGridLayout(self.central_widget)

        # Поле для ввода количества циклов
        self.cycle_label = QLabel("Задайте количество циклов:", self)
        self.cycle_entry = QLineEdit(self)
        self.set_cycle_button = QPushButton("Задать циклы", self)
        self.set_cycle_button.clicked.connect(self.create_cycle_inputs)

        self.layout.addWidget(self.cycle_label, 0, 0)
        self.layout.addWidget(self.cycle_entry, 0, 1)
        self.layout.addWidget(self.set_cycle_button, 0, 2)

        # Поле для вывода логов
        self.log_text = QTextEdit(self)
        self.log_text.setReadOnly(True)
        self.layout.addWidget(self.log_text, 1, 0, 1, 3)

        # Поля для отображения текущей и установленной температуры
        self.current_temp_label = QLabel("Текущая температура: -", self)
        self.set_temp_label = QLabel("Установленная температура: -", self)
        self.pwm_label = QLabel("Установленный ШИМ: -", self)
        self.layout.addWidget(self.current_temp_label, 2, 0)
        self.layout.addWidget(self.set_temp_label, 2, 1)
        self.layout.addWidget(self.pwm_label, 2, 2)

        # Поля для ввода циклов
        self.cycle_temps = []
        self.cycle_times = []
        self.cycle_inputs_layout = QGridLayout()
        self.layout.addLayout(self.cycle_inputs_layout, 3, 0, 2, 3)

        

        # График
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_ylim(0, 100)
        self.ax.set_xlim(0, 10)
        self.ax.set_title('Температура со временем')
        self.ax.set_xlabel('Время (с)')
        self.ax.set_ylabel('Температура (°C)')
        self.ax.grid(True)

        self.layout.addWidget(self.canvas, 0, 3, 6, 1)  # График размещается справа, от 0 до 6 ряда

        # Инициализация текущей и установленной температуры
        self.current_temp = None
        self.set_temp = None
        self.last_pwm_command = None
        self.current_pwm = None

    def log(self, message):
        QMetaObject.invokeMethod(self.log_text, "append", Qt.QueuedConnection, Q_ARG(str, message))

    def find_arduino_port(self):
        ports = serial.tools.list_ports.comports()
        self.log("Доступные порты:")
        for port in ports:
            self.log(f"{port.device} - {port.description}")
        for port in ports:
            if "Bluetooth" in port.description:
                continue
            try:
                self.log(f"Проверка порта {port.device}...")
                ser = serial.Serial(port.device, 115200, timeout=1)
                ser.write(b':R000D000000!\n')  # Тестовая команда для идентификации Arduino
                time.sleep(2)  # Ожидание ответа от контроллера
                response = ser.readline().decode('utf-8').strip()
                self.log(f"Ответ от порта {port.device}: {response}")
                if response:  # Предполагается, что Arduino отвечает на тестовую команду
                    self.log(f"Arduino найдена на {port.device}")
                    return ser
                ser.close()
            except (OSError, serial.SerialException) as e:
                self.log(f"Ошибка при доступе к порту {port.device}: {e}")
        return None

    def create_cycle_inputs(self):
        # Очистка предыдущих циклов
        for i in reversed(range(self.cycle_inputs_layout.count())):
            widget = self.cycle_inputs_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        num_cycles = int(self.cycle_entry.text())
        self.cycle_temps = []
        self.cycle_times = []

        for i in range(num_cycles):
            temp_label = QLabel(f"Цикл {i+1} - Температура (°C):", self)
            temp_entry = QLineEdit(self)
            self.cycle_temps.append(temp_entry)

            time_label = QLabel(f"Время удержания (с):", self)
            time_entry = QLineEdit(self)
            self.cycle_times.append(time_entry)

            self.cycle_inputs_layout.addWidget(temp_label, i, 0)
            self.cycle_inputs_layout.addWidget(temp_entry, i, 1)
            self.cycle_inputs_layout.addWidget(time_label, i, 2)
            self.cycle_inputs_layout.addWidget(time_entry, i, 3)

        # Обновляем расположение кнопки "Установить параметры циклов"
        if hasattr(self, 'set_cycle_params_button'):
            self.set_cycle_params_button.deleteLater()

        self.set_cycle_params_button = QPushButton("Установить параметры циклов", self)
        self.set_cycle_params_button.clicked.connect(self.start_cycle_thread)
        self.layout.addWidget(self.set_cycle_params_button, 4 + num_cycles, 0, 1, 3)

    def start_cycle_thread(self):
        cycle_thread = threading.Thread(target=self.send_cycle_settings)
        cycle_thread.start()

    def send_cycle_settings(self):
        if not self.serial_conn:
            self.log("Ошибка: соединение с Arduino не установлено.")
            return

        num_cycles = len(self.cycle_temps)
        for i, (temp_entry, time_entry) in enumerate(zip(self.cycle_temps, self.cycle_times)):
            try:
                temperature = float(temp_entry.text())
                duration = int(time_entry.text())
                self.set_temp = temperature
                self.log(f"Цикл: Температура {temperature} °C, Время удержания {duration} с.")

                # Обновление установленной температуры
                QMetaObject.invokeMethod(self.set_temp_label, "setText", Qt.QueuedConnection, Q_ARG(str, f"Установленная температура: {temperature} °C"))

                # Отправка температуры
                target_temp_command = f":R003D{int(temperature*10):06d}!\n"
                self.serial_conn.write(target_temp_command.encode())
                self.log(f"Отправлено на Arduino: {target_temp_command}")

                # Ожидание небольшого времени перед отправкой команды на возобновление регулирования
                time.sleep(0.1)

                # Отправка команды на возобновление регулирования
                pwm_command = ":R004D000001!"
                self.serial_conn.write(pwm_command.encode())
                self.log(f"Отправлено на Arduino: {pwm_command}")

                # Ожидание заданного времени удержания
                time.sleep(duration)

                # Если это последний цикл, отправляем команду на прекращение регулирования
                if i == num_cycles - 1:
                    self.stop_regulation()

            except (ValueError, serial.SerialException) as e:
                self.log(f"Ошибка отправки данных на Arduino: {e}")
                break

    def stop_regulation(self):
        stop_command = ":R004D000000!\n"  # Команда на прекращение регулирования
        try:
            self.serial_conn.write(stop_command.encode())
            self.log(f"Отправлено на Arduino: {stop_command}")
        except serial.SerialException as e:
            self.log(f"Ошибка отправки данных на Arduino: {e}")

    def read_from_arduino(self):
        if not self.serial_conn:
            self.log("Ошибка: соединение с Arduino не установлено.")
            return
        buffer = ""
        while True:
            try:
                in_waiting = self.serial_conn.in_waiting
                if in_waiting > 0:
                    buffer += self.serial_conn.read(in_waiting).decode('utf-8')
                    if '!' in buffer:  # Используем '!' как разделитель сообщений
                        lines = buffer.split('!')
                        for line in lines[:-1]:
                            line = line.strip()  # Удаляем пробелы
                            if line:
                                self.parse_arduino_message(line)
                        buffer = lines[-1]
            except serial.SerialException as e:
                self.log(f"Ошибка чтения данных с Arduino: {e}")
                break

    def parse_arduino_message(self, message):
        if message.startswith(":R001D"):
            temp_data = int(message[7:13])
            self.current_temp = temp_data / 10.0
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            self.xdata.append(elapsed_time)
            self.ydata.append(self.current_temp)
            QMetaObject.invokeMethod(self.current_temp_label, "setText", Qt.QueuedConnection, Q_ARG(str, f"Текущая температура: {self.current_temp} °C"))
            self.log_temperature(elapsed_time, self.current_temp, self.current_pwm)
            self.adjust_pwm()
        elif message.startswith(":R002D"):
            self.current_pwm = int(message[8:13])
            QMetaObject.invokeMethod(self.pwm_label, "setText", Qt.QueuedConnection, Q_ARG(str, f"Установленный ШИМ: {self.current_pwm}"))

    def log_temperature(self, time_elapsed, temperature, pwm):
        if self.log_file:
            timestamp = time.strftime("%H:%M:%S", time.gmtime(time_elapsed))
            log_line = f"{timestamp}\t{temperature}\t{pwm}\n"
            self.log_file.write(log_line)
            self.log_file.flush()

    def adjust_pwm(self):
        if self.current_temp is None or self.set_temp is None:
            return

        pwm_command = ":R004D000001!"

        if pwm_command != self.last_pwm_command:
            try:
                self.serial_conn.write(pwm_command.encode())
                self.log(f"Отправлено на Arduino: {pwm_command}")
                self.last_pwm_command = pwm_command
            except serial.SerialException as e:
                self.log(f"Ошибка отправки данных на Arduino: {e}")
                
    def update_plot(self):
    # Проверяем, что длины xdata и ydata совпадают
        if len(self.xdata) == len(self.ydata):
            self.ax.clear()
            self.ax.set_ylim(0, 100)
            self.ax.set_xlim(0, max(self.xdata) + 1 if self.xdata else 10)
            self.ax.set_title('Температура со временем')
            self.ax.set_xlabel('Время (с)')
            self.ax.set_ylabel('Температура (°C)')
            self.ax.grid(True)
            self.ax.plot(self.xdata, self.ydata, 'r-')
            self.canvas.draw()
        else:
            # Логирование или игнорирование, если размеры массивов не совпадают
            print(f"Warning: xdata and ydata have different lengths: {len(self.xdata)} vs {len(self.ydata)}")

    # def update_plot(self):
    #     self.ax.clear()
    #     self.ax.set_ylim(0, 100)
    #     self.ax.set_xlim(0, max(self.xdata) + 1 if self.xdata else 10)
    #     self.ax.set_title('Температура со временем')
    #     self.ax.set_xlabel('Время (с)')
    #     self.ax.set_ylabel('Температура (°C)')
    #     self.ax.grid(True)
    #     self.ax.plot(self.xdata, self.ydata, 'r-')
    #     self.canvas.draw()

    def closeEvent(self, event):
        if self.serial_conn:
            self.serial_conn.close()
        if self.log_file:
            self.log_file.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ArduinoGUI()
    gui.show()
    sys.exit(app.exec_())


    