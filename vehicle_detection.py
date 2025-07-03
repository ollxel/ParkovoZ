import cv2
import numpy as np
import os
import time
import argparse

# Настройка аргументов командной строки
parser = argparse.ArgumentParser(description='Vehicle Detection System')
parser.add_argument('--input', type=str, required=True, help='Path to input image or video')
parser.add_argument('--output', type=str, default='output', help='Output directory for results')
parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
parser.add_argument('--nms', type=float, default=0.4, help='NMS threshold')
parser.add_argument('--size', type=int, default=416, help='Input size for network (320, 416 or 512)')
parser.add_argument('--fps', type=int, default=10, help='Target FPS for video processing')
args = parser.parse_args()

# Создаем директорию для результатов
os.makedirs(args.output, exist_ok=True)

# Загрузка модели YOLOv4-tiny
print("[INFO] Загрузка модели YOLOv4-tiny...")
net = cv2.dnn.readNetFromDarknet("yolov4-tiny.cfg", "yolov4-tiny.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Загрузка названий классов
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# ID классов для транспорта
VEHICLE_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Получение выходных слоев
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


def detect_vehicles(image):
    """Обнаружение транспортных средств на изображении"""
    height, width = image.shape[:2]

    # Создаем blob для нейросети
    blob = cv2.dnn.blobFromImage(
        image,
        1 / 255.0,
        (args.size, args.size),
        swapRB=True,
        crop=False
    )

    # Подаем blob в сеть
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Обработка результатов
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > args.confidence and class_id in VEHICLE_IDS:
                # Масштабируем координаты bounding box
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype("int")

                # Получаем координаты верхнего левого угла
                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))

                # Гарантируем, что координаты в пределах изображения
                x, y = max(0, x), max(0, y)
                w, h = min(width - x, int(w)), min(height - y, int(h))

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Применяем Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, args.confidence, args.nms)

    return boxes, confidences, class_ids, indices


def process_image(image_path):
    """Обработка изображения"""
    print(f"[INFO] Обработка изображения: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        print(f"[ERROR] Не удалось загрузить изображение: {image_path}")
        return

    # Измеряем время обработки
    start_time = time.time()
    boxes, confidences, class_ids, indices = detect_vehicles(image)
    processing_time = time.time() - start_time

    # Счетчик транспортных средств
    vehicle_count = len(indices) if len(indices) > 0 else 0

    # Рисуем bounding boxes
    if vehicle_count > 0:
        for i in indices.flatten():
            (x, y, w, h) = boxes[i]

            # Рисуем прямоугольник
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Добавляем подпись
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.putText(image, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Сохраняем результат
    filename = os.path.basename(image_path)
    output_path = os.path.join(args.output, f"detected_{filename}")
    cv2.imwrite(output_path, image)

    # Выводим статистику
    print(f"[РЕЗУЛЬТАТ] Обнаружено транспортных средств: {vehicle_count}")
    print(f"[РЕЗУЛЬТАТ] Время обработки: {processing_time:.3f} сек")
    print(f"[РЕЗУЛЬТАТ] Результат сохранен в: {output_path}")

    # Показываем результат
    cv2.imshow("Vehicle Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(video_path):
    """Обработка видео"""
    print(f"[INFO] Обработка видео: {video_path}")

    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Не удалось открыть видео: {video_path}")
        return

    # Получаем свойства видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Создаем VideoWriter для сохранения результата
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_path = os.path.join(args.output, "detected_video.avi")
    out = cv2.VideoWriter(output_path, fourcc, args.fps, (width, height))

    print(f"[INFO] Разрешение: {width}x{height}")
    print(f"[INFO] Частота кадров: {fps:.2f}")
    print(f"[INFO] Всего кадров: {total_frames}")
    print(f"[INFO] Целевая частота обработки: {args.fps} FPS")

    frame_count = 0
    start_time = time.time()
    skip_frames = max(1, int(fps / args.fps))  # Пропускаем кадры для достижения целевого FPS

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Пропускаем кадры для достижения целевого FPS
        if frame_count % skip_frames != 0:
            continue

        # Обрабатываем каждый N-й кадр
        boxes, confidences, class_ids, indices = detect_vehicles(frame)
        vehicle_count = len(indices) if len(indices) > 0 else 0

        # Рисуем bounding boxes
        if vehicle_count > 0:
            for i in indices.flatten():
                (x, y, w, h) = boxes[i]

                # Рисуем прямоугольник
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Добавляем подпись
                label = f"{classes[class_ids[i]]}"
                cv2.putText(frame, label, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Добавляем статистику на кадр
        elapsed_time = time.time() - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"ТС: {vehicle_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Кадр: {frame_count}/{total_frames}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Сохраняем кадр и показываем
        out.write(frame)
        cv2.imshow("Vehicle Detection", frame)

        # Выход по ESC
        if cv2.waitKey(1) == 27:
            break

    # Завершение работы
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Выводим статистику
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0

    print(f"[РЕЗУЛЬТАТ] Обработка завершена!")
    print(f"[РЕЗУЛЬТАТ] Всего обработано кадров: {frame_count}")
    print(f"[РЕЗУЛЬТАТ] Средний FPS: {avg_fps:.1f}")
    print(f"[РЕЗУЛЬТАТ] Результат сохранен в: {output_path}")


def main():
    """Основная функция"""
    # Определяем тип файла по расширению
    input_path = args.input
    ext = os.path.splitext(input_path)[1].lower()

    # Обработка в зависимости от типа файла
    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        process_image(input_path)
    elif ext in ['.mp4', '.avi', '.mov']:
        process_video(input_path)
    else:
        print(f"[ERROR] Неподдерживаемый формат файла: {ext}")


if __name__ == "__main__":
    main()
