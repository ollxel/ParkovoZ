import cv2
import numpy as np
import os
import time

# Configuration files for YOLOv4 - используем YOLOv4-tiny для лучшей производительности
config_path = "yolov4-tiny.cfg"
weights_path = "yolov4-tiny.weights"
names_path = "coco.names"

# Проверяем наличие файлов
for file_path in [config_path, weights_path, names_path]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required file not found: {file_path}. Please download from:\n"
                                f"- Config: https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg\n"
                                f"- Weights: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights\n"
                                f"- Names: https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names")

# Загружаем YOLO модель
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Загружаем названия классов
with open(names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Получаем выходные слои
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# ID классов для транспорта (автомобиль, мотоцикл, автобус, грузовик)
vehicle_ids = [2, 3, 5, 7]


def detect_vehicles(image):
    """Обнаруживает транспорт на изображении и возвращает результаты"""
    height, width = image.shape[:2]

    # Создаем blob и запускаем детекцию
    blob = cv2.dnn.blobFromImage(
        image,
        1 / 255.0,
        (320, 320),  # Уменьшаем размер для повышения производительности
        swapRB=True,
        crop=False
    )
    net.setInput(blob)
    start = time.time()
    outputs = net.forward(output_layers)
    end = time.time()
    # print(f"Detection time: {end-start:.3f}s")

    # Обрабатываем результаты
    boxes = []
    confidences = []
    class_ids = []
    confidence_threshold = 0.5  # Порог уверенности
    nms_threshold = 0.4  # Порог для NMS

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold and class_id in vehicle_ids:
                # Масштабируем координаты bounding box'а
                box = detection[0:4] * np.array([width, height, width, height])
                (cx, cy, w, h) = box.astype("int")
                x = int(cx - (w / 2))
                y = int(cy - (h / 2))

                # Гарантируем, что координаты не выходят за границы изображения
                x, y, w, h = max(0, x), max(0, y), min(width, int(w)), min(height, int(h))

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Применяем Non-Maximum Suppression
    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    else:
        indices = []

    return boxes, confidences, class_ids, indices


def process_image(image_path):
    """Обрабатывает изображение и показывает результат"""
    if not os.path.exists(image_path):
        print(f"Ошибка: Файл изображения не найден: {image_path}")
        return

    print(f"Обработка изображения: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка: Не удалось прочитать изображение: {image_path}")
        return

    # Детектируем транспорт
    boxes, confidences, class_ids, indices = detect_vehicles(image)

    # Рисуем bounding box'ы и считаем транспорт
    vehicle_count = len(indices)
    print(f"Обнаружено транспортных средств: {vehicle_count}")

    if vehicle_count > 0:
        for i in indices:
            i = i[0] if isinstance(i, np.ndarray) else i
            (x, y, w, h) = boxes[i]
            # Рисуем прямоугольник
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Добавляем подпись
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.putText(image, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Сохраняем и показываем результат
    output_path = os.path.splitext(image_path)[0] + "_detected.jpg"
    cv2.imwrite(output_path, image)
    print(f"Результат сохранен: {output_path}")

    # Показываем изображение
    cv2.imshow('Vehicle Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(video_path):
    """Обрабатывает видеофайл с зацикливанием и контролем FPS"""
    if not os.path.exists(video_path):
        print(f"Ошибка: Видеофайл не найден: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео: {video_path}")
        return

    # Получаем свойства видео
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Целевой FPS - минимум 25 кадров/сек
    target_fps = 25.0
    frame_delay = max(1, int(1000 / target_fps))

    print(f"Обработка видео: {video_path}")
    print(f"Исходный FPS: {original_fps:.2f}, Целевой FPS: {target_fps}")

    frame_count = 0
    detection_frame_count = 0
    start_time = time.time()
    last_detection_time = start_time

    while True:
        ret, frame = cap.read()
        frame_count += 1

        # Перезапускаем видео по окончании
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            continue

        # Обрабатываем каждый 2-й кадр для повышения производительности
        process_frame = frame_count % 2 == 0

        vehicle_count = 0
        if process_frame:
            detection_frame_count += 1
            boxes, confidences, class_ids, indices = detect_vehicles(frame)
            vehicle_count = len(indices)

            # Рисуем bounding box'ы
            if vehicle_count > 0:
                for i in indices:
                    i = i[0] if isinstance(i, np.ndarray) else i
                    (x, y, w, h) = boxes[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f"{classes[class_ids[i]]}"
                    cv2.putText(frame, label, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Рассчитываем FPS
        elapsed_time = time.time() - start_time
        current_fps = detection_frame_count / elapsed_time if elapsed_time > 0 else 0

        # Выводим информацию о количестве каждую секунду
        if time.time() - last_detection_time >= 1.0:
            print(f"Кадр: {frame_count}, ТС: {vehicle_count}, FPS: {current_fps:.1f}")
            last_detection_time = time.time()

        # Отображаем информацию на кадре
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"ТС: {vehicle_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Показываем кадр
        cv2.imshow('Vehicle Detection', frame)

        # Выход по ESC
        key = cv2.waitKey(frame_delay)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    """Основная функция для обработки пользовательского ввода"""
    print("=" * 50)
    print("Система обнаружения транспортных средств")
    print("=" * 50)

    while True:
        choice = input("Обработать видео или изображение? (введите 'v' для видео, 'i' для изображения): ").lower()
        if choice in ['v', 'video']:
            video_path = input("Введите путь к видеофайлу: ")
            process_video(video_path)
            break
        elif choice in ['i', 'image']:
            image_path = input("Введите путь к файлу изображения: ")
            process_image(image_path)
            break
        else:
            print("Некорректный выбор. Пожалуйста, введите 'v' или 'i'.")


if __name__ == "__main__":
    main()