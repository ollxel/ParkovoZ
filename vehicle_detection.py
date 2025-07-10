import cv2
import numpy as np
import os
import time
import argparse
import requests
from datetime import datetime
import threading
import queue
import websockets
import asyncio
import json
import logging
from ultralytics import YOLO  # Добавляем импорт для YOLOv8

# Настройка аргументов командной строки
parser = argparse.ArgumentParser(description='Parking Lot Monitoring System')
parser.add_argument('--url', type=str, required=True, help='Camera stream URL')
parser.add_argument('--output', type=str, default='output', help='Output directory for results')
parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
parser.add_argument('--nms', type=float, default=0.4, help='NMS threshold')
parser.add_argument('--size', type=int, default=640, help='Input size for network')  # Размер по умолчанию для YOLOv8
parser.add_argument('--fps', type=int, default=15, help='Target FPS for video processing')
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'], help='Device for inference')
parser.add_argument('--spots', type=str, default='parking_spots.txt', help='File to save/load parking spots')
parser.add_argument('--ws-port', type=int, default=9000, help='WebSocket server port')
parser.add_argument('--model', type=str, default='yolov8s.pt', help='Path to YOLOv8 model')  # Новый аргумент для модели
args = parser.parse_args()

# Создаем директорию для результатов
os.makedirs(args.output, exist_ok=True)

# Настройка логгера
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка модели YOLOv8
logging.info(f"Загрузка модели YOLOv8: {args.model}")
model = YOLO(args.model)

# Настройка устройства для вычислений
device = 'cuda' if args.device == 'gpu' else 'cpu'
model.to(device)
logging.info(f"Используется {device.upper()} для вычислений")

# Загрузка названий классов
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# ID классов для транспорта
VEHICLE_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Статус записи видео
is_recording = False
video_writer = None
recording_start_time = None
MAX_RECORDING_MINUTES = 5

# Очередь для многопоточной загрузки кадров
frame_queue = queue.Queue(maxsize=2)
stop_event = threading.Event()

# Состояние парковочных мест
parking_spots = []  # список кортежей (x, y, status): 0=свободно, 1=занято
marking_mode = False
current_spot_id = 0

# WebSocket клиенты и event loop
connected_clients = set()
client_lock = threading.Lock()
ws_loop = None

async def websocket_handler(websocket, path):
    with client_lock:
        connected_clients.add(websocket)
    logging.info(f"Новый клиент подключен. Всего клиентов: {len(connected_clients)}")
    
    try:
        # Отправляем текущее состояние при подключении
        if parking_spots:
            spot_states = [s[2] for s in parking_spots]
            await websocket.send(json.dumps({
                'type': 'parking_data',
                'data': spot_states,
                'timestamp': datetime.now().isoformat()
            }))
        
        # Ожидаем закрытия соединения
        await websocket.wait_closed()
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
    finally:
        with client_lock:
            connected_clients.discard(websocket)

async def send_to_clients(data):
    if connected_clients:
        for client in connected_clients.copy():
            try:
                await client.send(json.dumps(data))
            except websockets.exceptions.ConnectionClosed:
                logging.warning("Клиент отключился")
                with client_lock:
                    connected_clients.discard(client)

def start_websocket_server():
    global ws_loop
    ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(ws_loop)
    
    async def start():
        server = await websockets.serve(
            websocket_handler, 
            "0.0.0.0",  # Слушаем все интерфейсы
            args.ws_port
        )
        logging.info(f"WebSocket сервер запущен на порту {args.ws_port}")
        await server.wait_closed()
    
    ws_loop.run_until_complete(start())
    ws_loop.run_forever()

def image_loader(url):
    """Поток для загрузки изображений"""
    last_frame = None
    while not stop_event.is_set():
        try:
            response = requests.get(url, timeout=2, stream=True)
            response.raise_for_status()
            img_array = np.frombuffer(response.content, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is not None:
                last_frame = frame
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                frame_queue.put(frame.copy())
            else:
                if last_frame is not None:
                    frame_queue.put(last_frame.copy())
        except:
            if last_frame is not None:
                frame_queue.put(last_frame.copy())
        time.sleep(0.05)

def detect_vehicles(image):
    """Обнаружение транспортных средств на изображении с помощью YOLOv8"""
    height, width = image.shape[:2]
    
    # Выполняем детекцию
    results = model.predict(
        image,
        imgsz=args.size,
        conf=args.confidence,
        iou=args.nms,
        device=device,
        verbose=False
    )
    
    boxes = []
    confidences = []
    class_ids = []
    centers = []

    # Обработка результатов
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                # Фильтрация по классу транспортных средств
                if int(box.cls) in VEHICLE_IDS:
                    # Координаты bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Гарантируем, что координаты в пределах изображения
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    
                    w = x2 - x1
                    h = y2 - y1
                    
                    # Центр bounding box
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    boxes.append([x1, y1, w, h])
                    centers.append((center_x, center_y))
                    confidences.append(float(box.conf))
                    class_ids.append(int(box.cls))

    # Для совместимости с оригинальным интерфейсом
    indices = list(range(len(boxes))) if boxes else []
    
    return boxes, confidences, class_ids, indices, centers

def create_video_capture(url):
    """Создает объект VideoCapture для потокового видео"""
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    return cap

def save_parking_spots(filename):
    """Сохраняет координаты парковочных мест в файл"""
    with open(filename, 'w') as f:
        for spot in parking_spots:
            f.write(f"{spot[0]},{spot[1]},{spot[2]}\n")
    logging.info(f"Сохранено {len(parking_spots)} парковочных мест в {filename}")

def load_parking_spots(filename):
    """Загружает координаты парковочных мест из файла"""
    global parking_spots
    if os.path.exists(filename):
        parking_spots = []
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    x, y, status = int(parts[0]), int(parts[1]), int(parts[2])
                    parking_spots.append((x, y, status))
        logging.info(f"Загружено {len(parking_spots)} парковочных мест из {filename}")
        return True
    return False

def mouse_callback(event, x, y, flags, param):
    """Обработчик событий мыши для разметки парковочных мест"""
    global marking_mode, parking_spots, frame_copy
    
    if marking_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            # Добавляем свободное место (зеленое)
            parking_spots.append((x, y, 0))
            logging.info(f"Добавлено свободное место {len(parking_spots)}: ({x}, {y})")
            # Рисуем точку на изображении
            cv2.circle(frame_copy, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(frame_copy, str(len(parking_spots)), (x+10, y+10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Mark Parking Spots", frame_copy)
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Добавляем занятое место (красное)
            parking_spots.append((x, y, 1))
            logging.info(f"Добавлено занятое место {len(parking_spots)}: ({x}, {y})")
            # Рисуем точку на изображении
            cv2.circle(frame_copy, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(frame_copy, str(len(parking_spots)), (x+10, y+10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Mark Parking Spots", frame_copy)

def mark_parking_spots(frame):
    """Режим ручной разметки парковочных мест"""
    global marking_mode, frame_copy, parking_spots
    
    marking_mode = True
    frame_copy = frame.copy()
    
    # Создаем окно для разметки
    cv2.namedWindow("Mark Parking Spots")
    cv2.setMouseCallback("Mark Parking Spots", mouse_callback, param=frame)
    
    # Инструкция
    logging.info("\n=== РЕЖИМ РАЗМЕТКИ ПАРКОВОЧНЫХ МЕСТ ===")
    logging.info("ЛКМ - добавить свободное место (зеленое)")
    logging.info("ПКМ - добавить занятое место (красное)")
    logging.info("'z' - удалить последнее добавленное место")
    logging.info("'s' - сохранить разметку")
    logging.info("'c' - отменить и выйти")
    
    while True:
        cv2.imshow("Mark Parking Spots", frame_copy)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            if parking_spots:
                save_parking_spots(args.spots)
                marking_mode = False
                cv2.destroyWindow("Mark Parking Spots")
                return True
            else:
                logging.warning("Нет парковочных мест для сохранения!")
        
        elif key == ord('c'):
            parking_spots = []
            marking_mode = False
            cv2.destroyWindow("Mark Parking Spots")
            logging.info("Разметка отменена")
            return False
        
        elif key == ord('z'):
            # Удаляем последнее добавленное место
            if parking_spots:
                removed = parking_spots.pop()
                logging.info(f"Удалено парковочное место: {removed}")
                # Перерисовываем изображение
                frame_copy = frame.copy()
                for i, (x, y, status) in enumerate(parking_spots):
                    color = (0, 255, 0) if status == 0 else (0, 0, 255)
                    cv2.circle(frame_copy, (x, y), 8, color, -1)
                    cv2.putText(frame_copy, str(i+1), (x+10, y+10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.imshow("Mark Parking Spots", frame_copy)

def process_stream():
    """Обработка потока с камеры в реальном времени"""
    global is_recording, video_writer, recording_start_time, parking_spots, marking_mode

    logging.info(f"Запуск мониторинга парковки: {args.url}")

    # Автоматическое определение типа потока
    is_video_stream = False
    cap = None
    last_success_time = time.time()

    # Попытка инициализировать как видеопоток
    if "mjpg" in args.url.lower() or "stream" in args.url.lower():
        try:
            cap = create_video_capture(args.url)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    is_video_stream = True
                    logging.info("Режим: MJPEG видеопоток")
                else:
                    cap.release()
                    cap = None
        except:
            pass

    # Запускаем поток загрузки изображений для статичного режима
    if not is_video_stream:
        logging.info("Режим: Статичное изображение (многопоточная загрузка)")
        loader_thread = threading.Thread(target=image_loader, args=(args.url,), daemon=True)
        loader_thread.start()

    # Создаем окно для отображения
    cv2.namedWindow("Parking Lot Monitoring", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Parking Lot Monitoring", 1000, 700)

    frame_count = 0
    total_vehicles = 0
    start_time = time.time()
    last_frame_time = time.time()
    frame = None
    reconnect_attempts = 0
    last_stat_update = time.time()
    stats_interval = 0.5
    current_fps = 0
    vehicle_count = 0
    first_frame_processed = False

    try:
        while True:
            current_time = time.time()
            # Контроль FPS
            elapsed = current_time - last_frame_time
            sleep_time = (1.0 / args.fps) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            last_frame_time = time.time()
            frame_count += 1

            # Получаем кадр в зависимости от типа потока
            if is_video_stream and cap is not None:
                ret, frame = cap.read()
                if not ret or frame is None:
                    # Проблемы с чтением кадра
                    if current_time - last_success_time > 5:
                        logging.warning("Ошибка чтения видеопотока. Попытка переподключения...")
                        cap.release()
                        cap = create_video_capture(args.url)
                        reconnect_attempts += 1

                        # Если переподключение не удалось 5 раз, переключаем режим
                        if reconnect_attempts > 5 or not cap.isOpened():
                            logging.warning("Не удалось восстановить видеопоток. Переключение на режим статичного изображения")
                            is_video_stream = False
                            loader_thread = threading.Thread(target=image_loader, args=(args.url,), daemon=True)
                            loader_thread.start()
                    continue
                last_success_time = current_time
                reconnect_attempts = 0
            else:
                try:
                    frame = frame_queue.get_nowait()
                except queue.Empty:
                    frame = None

            if frame is None:
                # Показываем последний успешный кадр, если он есть
                if 'last_good_frame' in locals():
                    frame = last_good_frame
                else:
                    continue

            # Кэшируем хороший кадр
            last_good_frame = frame.copy()
            
            # Автоматический вход в режим разметки при первом кадре, если файл не найден
            if not first_frame_processed:
                first_frame_processed = True
                if not os.path.exists(args.spots):
                    logging.info("Файл с разметкой не найден, запуск режима разметки...")
                    if mark_parking_spots(frame):
                        logging.info("Разметка парковочных мест сохранена")
                    else:
                        logging.info("Разметка парковочных мест отменена, выход")
                        break

            # Обнаруживаем транспортные средства
            boxes, confidences, class_ids, indices, centers = detect_vehicles(frame)
            vehicle_count = len(indices) if isinstance(indices, list) else 0
            total_vehicles += vehicle_count

            # Определяем занятость парковочных мест
            occupied_spots = [False] * len(parking_spots)
            
            # Проверяем для каждого парковочного места
            for i, (x, y, _) in enumerate(parking_spots):
                for center in centers:
                    # Проверяем расстояние от центра машины до точки парковки
                    distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                    # Если расстояние меньше порога (15 пикселей), считаем место занятым
                    if distance < 15:
                        occupied_spots[i] = True
                        break

            # Подготавливаем данные для отправки
            spot_states = [1 if occupied else 0 for occupied in occupied_spots]
            free_count = spot_states.count(0)
            occupied_count = spot_states.count(1)
            
            # Отправляем данные через WebSocket
            if ws_loop and ws_loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    send_to_clients({
                        'type': 'parking_data',
                        'data': spot_states,
                        'free': free_count,
                        'occupied': occupied_count,
                        'timestamp': datetime.now().isoformat()
                    }), 
                    ws_loop
                )

            # Рисуем bounding boxes (только если есть транспорт)
            if vehicle_count > 0:
                for i, box in enumerate(boxes):
                    (x, y, w, h) = box
                    # Рисуем прямоугольник
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Добавляем подпись
                    label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
                    cv2.putText(frame, label, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Рисуем парковочные места и их статус
            free_count = 0
            for i, (x, y, initial_status) in enumerate(parking_spots):
                # Определяем цвет на основе текущей занятости
                if occupied_spots[i]:
                    color = (0, 0, 255)  # Красный - занято
                else:
                    color = (0, 255, 0)  # Зеленый - свободно
                    free_count += 1
                
                # Рисуем точку с текущим статусом
                cv2.circle(frame, (x, y), 8, color, -1)
                cv2.putText(frame, str(i+1), (x+10, y+10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Обновляем статистику с заданным интервалом
            current_time = time.time()
            if current_time - last_stat_update > stats_interval:
                elapsed_time = current_time - start_time
                current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                frame_count = 0
                start_time = current_time
                last_stat_update = current_time

            # Добавляем статистику на кадр
            stream_type = "Видеопоток" if is_video_stream else "Статичное изображение"
            stats = [
                f"Парковка: {free_count}/{len(parking_spots)} св.",
                f"ТС: {vehicle_count} | Всего: {total_vehicles}",
                f"FPS: {current_fps:.1f} (Цель: {args.fps})",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                f"Тип: {stream_type}",
                f"Размер: {args.size}px | Устр: {args.device.upper()}",
                f"Модель: {os.path.basename(args.model)}"
            ]

            for i, stat in enumerate(stats):
                cv2.putText(frame, stat, (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Статус записи
            if is_recording:
                cv2.putText(frame, "REC", (frame.shape[1] - 100, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                recording_time = time.time() - recording_start_time
                mins, secs = divmod(int(recording_time), 60)
                cv2.putText(frame, f"{mins:02d}:{secs:02d}", (frame.shape[1] - 100, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Запись в видеофайл при активации
            if is_recording and video_writer is not None:
                video_writer.write(frame)

                # Проверка максимального времени записи
                if time.time() - recording_start_time > MAX_RECORDING_MINUTES * 60:
                    stop_recording()

            # Отображаем результат
            cv2.imshow("Parking Lot Monitoring", frame)

            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Сохранение текущего снимка
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(args.output, f"parking_snapshot_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                logging.info(f"Снимок сохранен: {filename}")
            elif key == ord('r'):
                # Начать/остановить запись
                if is_recording:
                    stop_recording()
                else:
                    start_recording(frame)
            elif key == ord('m'):
                # Переключение режима вручную
                is_video_stream = not is_video_stream
                if cap is not None:
                    cap.release()
                    cap = None
                if is_video_stream:
                    logging.info("Переключено на режим видеопотока")
                    cap = create_video_capture(args.url)
                else:
                    logging.info("Переключено на режим статичного изображения")
                    stop_event.clear()
                    loader_thread = threading.Thread(target=image_loader, args=(args.url,), daemon=True)
                    loader_thread.start()
            elif key == ord('p'):
                # Вход в режим разметки парковочных мест
                if mark_parking_spots(frame):
                    logging.info("Разметка парковочных мест сохранена")
                else:
                    logging.info("Разметка парковочных мест отменена")

    except KeyboardInterrupt:
        logging.info("Программа остановлена пользователем")
    except Exception as e:
        logging.error(f"Критическая ошибка: {e}")
    finally:
        stop_event.set()
        if cap is not None and cap.isOpened():
            cap.release()
        if is_recording:
            stop_recording()
        cv2.destroyAllWindows()


def start_recording(frame):
    """Начинает запись видео"""
    global is_recording, video_writer, recording_start_time

    if not is_recording:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.output, f"parking_recording_{timestamp}.avi")

        # Получаем размеры кадра
        height, width = frame.shape[:2]

        # Создаем VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(output_path, fourcc, args.fps, (width, height))

        if video_writer.isOpened():
            is_recording = True
            recording_start_time = time.time()
            logging.info(f"Начата запись видео: {output_path}")
        else:
            logging.error("Не удалось создать видеофайл для записи")


def stop_recording():
    """Останавливает запись видео"""
    global is_recording, video_writer

    if is_recording and video_writer is not None:
        video_writer.release()
        is_recording = False
        logging.info("Запись видео остановлена")


def main():
    """Основная функция"""
    print("=" * 70)
    print(f"Система мониторинга парковки | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    print("Инструкция:")
    print("- Нажмите 's' для сохранения снимка")
    print("- Нажмите 'r' для начала/остановки записи видео")
    print("- Нажмите 'm' для переключения режима (поток/изображение)")
    print("- Нажмите 'p' для разметки парковочных мест")
    print("- Нажмите 'q' для выхода из программы")
    print(f"\nИспользуемая камера: {args.url}")
    print(f"Модель: YOLOv8 | Размер: {args.size}px | Устройство: {args.device.upper()}")
    print("=" * 70)

    # Проверка наличия файла модели
    if not os.path.exists(args.model):
        print(f"[ERROR] Файл модели YOLOv8 не найден: {args.model}")
        print("Скачайте предобученные модели с https://github.com/ultralytics/ultralytics")
        return

    # Запуск WebSocket сервера в отдельном потоке
    ws_thread = threading.Thread(target=start_websocket_server, daemon=True)
    ws_thread.start()

    # Запуск обработки потока
    process_stream()


if __name__ == "__main__":
    main()
