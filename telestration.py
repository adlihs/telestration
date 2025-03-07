import cv2
import numpy as np
import argparse
import os
from scipy.spatial import ConvexHull

# Global variables
drawing = False  # True if the mouse is pressed
ix, iy = -1, -1  # Initial coordinates
fx, fy = -1, -1  # Final coordinates
paused = False  # Video pause state
current_frame = None  # Current frame
drawings = []  # List to store drawings
temp_frame = None  # Temporary frame for dynamic drawing
draw_mode = 'line'  # Current drawing mode (line, circle, arrow, dashed_arrow, rectangle, spotlight, text)
line_style = 'solid'  # Line style (solid, dashed)
selected_index = -1  # Index of the selected element (-1 if no selection)
dragging = False  # True if an element is being dragged
text_input = ""  # User-entered text
text_position = (-1, -1)  # Text position in the frame
selected_corner = -1  # Selected corner of the rectangle (-1 if no selection)
circles_for_hull = []  # List of circle centers for convex hull
polygon_points = []  # Lista para almacenar los puntos del polígono


# Función para dibujar un polígono con relleno transparente y líneas horizontales
def draw_polygon_with_horizontal_lines(img, points, color=(0, 255, 255), alpha=0.3, line_spacing=10):
    # Convertir los puntos a un array de numpy
    pts = np.array(points, dtype=np.int32)

    # Crear una capa transparente para el relleno del polígono
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], color)  # Rellenar el polígono
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)  # Aplicar transparencia

    # Dibujar el borde del polígono
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

    # Calcular las coordenadas para las líneas horizontales
    y_min = min(p[1] for p in points)
    y_max = max(p[1] for p in points)

    # Dibujar líneas horizontales dentro del polígono
    for y in range(y_min, y_max, line_spacing):
        # Encontrar los puntos de intersección de la línea horizontal con el polígono
        intersections = []
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            if min(y1, y2) <= y <= max(y1, y2):
                if y1 != y2:
                    x = int(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
                    intersections.append(x)

        # Dibujar la línea horizontal si hay dos intersecciones
        if len(intersections) == 2:
            cv2.line(img, (intersections[0], y), (intersections[1], y), color, 1)


# Function to draw an inverted triangle
def draw_inverted_triangle(img, center, size=10, color=(255, 255, 255), alpha=0.3):
    x, y = center
    pt1 = (x, y - size)  # Top point
    pt2 = (x - size, y + size)  # Bottom left point
    pt3 = (x + size, y + size)  # Bottom right point
    triangle_pts = np.array([pt1, pt2, pt3], dtype=np.int32)

    # Create a transparent overlay for the triangle
    overlay = img.copy()
    cv2.fillPoly(overlay, [triangle_pts], color)  # Fill the triangle
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)  # Apply transparency

    # Draw the triangle border
    cv2.polylines(img, [triangle_pts], isClosed=True, color=color, thickness=2)


# Function to draw a dashed line
def draw_dashed_line(img, pt1, pt2, color, thickness, dash_length=10):
    dist = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
    dash_count = int(dist / dash_length)
    for i in range(dash_count):
        start = (int(pt1[0] + (pt2[0] - pt1[0]) * i / dash_count),
                 int(pt1[1] + (pt2[1] - pt1[1]) * i / dash_count))
        end = (int(pt1[0] + (pt2[0] - pt1[0]) * (i + 0.5) / dash_count),
               int(pt1[1] + (pt2[1] - pt1[1]) * (i + 0.5) / dash_count))
        cv2.line(img, start, end, color, thickness)


# Function to apply a spotlight effect
def apply_spotlight(img, center, radius, intensity=0.3):
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.circle(mask, center, radius, (255, 255, 255), -1)
    mask = cv2.GaussianBlur(mask, (0, 0), radius / 3)
    mask = mask.astype(np.float32) / 255.0
    inverse_mask = 1.0 - mask
    img_float = img.astype(np.float32) / 255.0
    spotlight = img_float * mask + img_float * inverse_mask * intensity
    return (spotlight * 255).astype(np.uint8)


# Function to check if a point is inside an element
def is_point_inside_element(x, y, element):
    if element[0] == 'line':
        (x1, y1), (x2, y2) = element[1], element[2]
        # Check if the point is near the line (using a minimum distance)
        dist = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        return dist < 5  # Tolerance of 5 pixels
    elif element[0] == 'circle':
        (cx, cy), (rx, ry) = element[1], element[2]
        # Check if the point is inside the ellipse
        return ((x - cx) ** 2 / rx ** 2) + ((y - cy) ** 2 / ry ** 2) <= 1
    elif element[0] == 'rectangle':
        (x1, y1), (x2, y2) = element[1], element[2]
        return x1 <= x <= x2 and y1 <= y <= y2
    elif element[0] == 'arrow' or element[0] == 'dashed_arrow':
        (x1, y1), (x2, y2) = element[1], element[2]
        # Check if the point is near the arrow line
        dist = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        return dist < 5  # Tolerance of 5 pixels
    elif element[0] == 'spotlight':
        (cx, cy), radius = element[1], element[2]
        return np.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= radius
    elif element[0] == 'text':
        (tx, ty), _ = element[1], element[2]
        # Check if the point is near the text position
        return abs(x - tx) < 50 and abs(y - ty) < 20  # Tolerance of 50x20 pixels
    return False


# Function to draw shapes with transparent fill
def draw_transparent_shape(img, shape, alpha=0.3):
    overlay = img.copy()
    if shape[0] == 'circle':
        center, (rx, ry) = shape[1], shape[2]
        cv2.ellipse(overlay, center, (rx, ry), 0, 0, 360, (0, 0, 255), -1)  # Fill
        cv2.ellipse(overlay, center, (rx, ry), 0, 0, 360, (0, 0, 255), 2)  # Border
    elif shape[0] == 'rectangle':
        (x1, y1), (x2, y2) = shape[1], shape[2]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), -1)  # Fill
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Border
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


# Function to draw the convex hull
def draw_convex_hull(img, points, color=(0, 255, 0), alpha=0.3):
    if len(points) >= 3:  # At least 3 points are needed for a convex hull
        hull = ConvexHull(points)
        hull_points = [points[i] for i in hull.vertices]
        hull_points = np.array(hull_points, dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [hull_points], color)  # Transparent fill
        cv2.polylines(overlay, [hull_points], True, color, 2)  # Border
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


# Function to save the current frame as an image
def save_frame_with_drawings(frame, drawings, output_folder="output"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, "frame_with_drawings.png")
    display_frame = frame.copy()

    # Dibujar todas las formas en el frame
    for shape in drawings:
        if shape[0] == 'line':
            if shape[3] == 'solid':
                cv2.line(display_frame, shape[1], shape[2], (0, 255, 0), 2)
            elif shape[3] == 'dashed':
                draw_dashed_line(display_frame, shape[1], shape[2], (0, 255, 0), 2)
        elif shape[0] == 'circle' or shape[0] == 'rectangle':
            draw_transparent_shape(display_frame, shape)
        elif shape[0] == 'spotlight':
            display_frame = apply_spotlight(display_frame, shape[1], shape[2])
        elif shape[0] == 'arrow':
            if shape[3] == 'solid':
                cv2.arrowedLine(display_frame, shape[1], shape[2], (255, 0, 0), 2)
            elif shape[3] == 'dashed':
                draw_dashed_line(display_frame, shape[1], shape[2], (255, 0, 0), 2)
                # Dibujar la punta de la flecha manualmente
                angle = np.arctan2(shape[2][1] - shape[1][1], shape[2][0] - shape[1][0])
                arrow_tip1 = (int(shape[2][0] - 15 * np.cos(angle + np.pi / 6)),
                              int(shape[2][1] - 15 * np.sin(angle + np.pi / 6)))
                arrow_tip2 = (int(shape[2][0] - 15 * np.cos(angle - np.pi / 6)),
                              int(shape[2][1] - 15 * np.sin(angle - np.pi / 6)))
                cv2.line(display_frame, shape[2], arrow_tip1, (255, 0, 0), 2)
                cv2.line(display_frame, shape[2], arrow_tip2, (255, 0, 0), 2)
        elif shape[0] == 'text':
            (tx, ty), text = shape[1], shape[2]
            cv2.putText(display_frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        elif shape[0] == 'triangle':
            draw_inverted_triangle(display_frame, shape[1], shape[2], (255, 255, 255), 0.3)
        elif shape[0] == 'polygon':
            # Dibujar el polígono con relleno transparente y líneas horizontales
            draw_polygon_with_horizontal_lines(display_frame, shape[1], color=(0, 255, 255), alpha=0.3)

    # Dibujar el convex hull si hay suficientes círculos
    if len(circles_for_hull) >= 3:
        hull = ConvexHull(np.array(circles_for_hull))
        hull_points = [circles_for_hull[i] for i in hull.vertices]
        hull_points = np.array(hull_points, dtype=np.int32)

        # Dibujar el convex hull en el display_frame
        overlay = display_frame.copy()
        cv2.fillPoly(overlay, [hull_points], (0, 255, 0))  # Relleno transparente
        cv2.polylines(overlay, [hull_points], True, (0, 255, 0), 2)  # Borde
        cv2.addWeighted(overlay, 0.3, display_frame, 1 - 0.3, 0, display_frame)

        # Calcular el área del convex hull
        hull_area = hull.area
        #area_text = f"Area del Convex Hull: {hull_area:.2f} px²"

        # Mostrar el área en la imagen
        #cv2.putText(display_frame, area_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Guardar el área en un archivo de texto
        #area_output_path = os.path.join(output_folder, "convex_hull_area.txt")
        #with open(area_output_path, "w") as f:
            #f.write(area_text)

        #print(f"Area del Convex Hull guardada en: {area_output_path}")

    # Guardar la imagen con los dibujos y el convex hull
    cv2.imwrite(output_path, display_frame)
    print(f"Frame guardado en: {output_path}")


# Function to check if a point is near a corner of the rectangle
def is_point_near_corner(x, y, rect):
    (x1, y1), (x2, y2) = rect[1], rect[2]
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    for i, (cx, cy) in enumerate(corners):
        if abs(x - cx) < 10 and abs(y - cy) < 10:  # Tolerance of 10 pixels
            return i  # Return the corner index
    return -1  # No corner found


# Mouse callback function
def draw_shape(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing, temp_frame, selected_index, dragging, text_position, selected_corner, circles_for_hull, polygon_points

    if event == cv2.EVENT_LBUTTONDOWN:
        # Verificar si se hizo clic en un elemento existente
        for i, element in enumerate(drawings):
            if is_point_inside_element(x, y, element):
                selected_index = i
                if element[0] == 'rectangle':
                    selected_corner = is_point_near_corner(x, y, element)
                dragging = True
                break
        else:
            # Si no se seleccionó ningún elemento, comenzar a dibujar
            if draw_mode == 'text':
                text_position = (x, y)
                text_input = input("Ingrese el texto: ")
                drawings.append(('text', text_position, text_input))
            elif draw_mode == 'triangle':
                # Dibujar un triángulo invertido en la posición del clic
                drawings.append(('triangle', (x, y), 10))  # 10 es el tamaño del triángulo
            elif draw_mode == 'polygon':
                # Almacenar los puntos del polígono
                if len(polygon_points) < 4:
                    polygon_points.append((x, y))
                    print(f"Punto {len(polygon_points)} registrado: ({x}, {y})")
                    if len(polygon_points) == 4:
                        # Dibujar el polígono cuando se tengan 4 puntos
                        drawings.append(('polygon', polygon_points.copy()))
                        polygon_points.clear()  # Limpiar la lista para un nuevo polígono
            else:
                drawing = True
                ix, iy = x, y
                fx, fy = x, y  # Inicializar las coordenadas finales

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging and selected_index != -1:
            # Mover el elemento seleccionado o deformar el rectángulo
            dx, dy = x - fx, y - fy
            element = drawings[selected_index]
            if element[0] == 'rectangle' and selected_corner != -1:
                # Deformar el rectángulo arrastrando una esquina
                (x1, y1), (x2, y2) = element[1], element[2]
                if selected_corner == 0:  # Esquina superior izquierda
                    x1, y1 = x, y
                elif selected_corner == 1:  # Esquina superior derecha
                    x2, y1 = x, y
                elif selected_corner == 2:  # Esquina inferior derecha
                    x2, y2 = x, y
                elif selected_corner == 3:  # Esquina inferior izquierda
                    x1, y2 = x, y
                drawings[selected_index] = ('rectangle', (x1, y1), (x2, y2))
            else:
                # Mover el elemento seleccionado
                if element[0] == 'line' or element[0] == 'arrow' or element[0] == 'dashed_arrow':
                    drawings[selected_index] = (element[0], (element[1][0] + dx, element[1][1] + dy),
                                                (element[2][0] + dx, element[2][1] + dy), *element[3:])
                elif element[0] == 'circle':
                    (cx, cy), (rx, ry) = element[1], element[2]
                    drawings[selected_index] = (element[0], (cx + dx, cy + dy), (rx, ry))
                elif element[0] == 'rectangle':
                    drawings[selected_index] = (element[0], (element[1][0] + dx, element[1][1] + dy),
                                                (element[2][0] + dx, element[2][1] + dy))
                elif element[0] == 'spotlight':
                    drawings[selected_index] = (element[0], (element[1][0] + dx, element[1][1] + dy), element[2])
                elif element[0] == 'text':
                    (tx, ty), text = element[1], element[2]
                    drawings[selected_index] = (element[0], (tx + dx, ty + dy), text)
                elif element[0] == 'triangle':
                    (tx, ty), size = element[1], element[2]
                    drawings[selected_index] = (element[0], (tx + dx, ty + dy), size)
            fx, fy = x, y
        elif drawing:
            fx, fy = x, y
            # Redibujar el frame temporal con la forma dinámica
            temp_frame = current_frame.copy()
            if draw_mode == 'line':
                if line_style == 'solid':
                    cv2.line(temp_frame, (ix, iy), (fx, fy), (0, 255, 0), 2)
                elif line_style == 'dashed':
                    draw_dashed_line(temp_frame, (ix, iy), (fx, fy), (0, 255, 0), 2)
            elif draw_mode == 'circle':
                rx = int(abs(fx - ix))  # Radio horizontal
                ry = int(abs(fy - iy) / 2)  # Radio vertical (mitad del radio horizontal)
                draw_transparent_shape(temp_frame, ('circle', ((ix + fx) // 2, (iy + fy) // 2), (rx, ry)))
            elif draw_mode == 'rectangle':
                # Ajustar la altura del rectángulo para que sea "acostado"
                width = abs(fx - ix)
                height = int(width / 2)  # Altura proporcional al ancho
                x1 = min(ix, fx)
                y1 = min(iy, fy) - height // 2
                x2 = max(ix, fx)
                y2 = max(iy, fy) + height // 2
                draw_transparent_shape(temp_frame, ('rectangle', (x1, y1), (x2, y2)))
            elif draw_mode == 'spotlight':
                radius = int(np.sqrt((fx - ix) ** 2 + (fy - iy) ** 2))
                temp_frame = apply_spotlight(temp_frame, (ix, iy), radius)
            elif draw_mode in ['arrow', 'dashed_arrow']:
                if line_style == 'solid':
                    cv2.arrowedLine(temp_frame, (ix, iy), (fx, fy), (255, 0, 0), 2)
                elif line_style == 'dashed':
                    draw_dashed_line(temp_frame, (ix, iy), (fx, fy), (255, 0, 0), 2)
                    # Dibujar la punta de la flecha manualmente
                    angle = np.arctan2(fy - iy, fx - ix)
                    arrow_tip1 = (int(fx - 15 * np.cos(angle + np.pi / 6)),
                                  int(fy - 15 * np.sin(angle + np.pi / 6)))
                    arrow_tip2 = (int(fx - 15 * np.cos(angle - np.pi / 6)),
                                  int(fy - 15 * np.sin(angle - np.pi / 6)))
                    cv2.line(temp_frame, (fx, fy), arrow_tip1, (255, 0, 0), 2)
                    cv2.line(temp_frame, (fx, fy), arrow_tip2, (255, 0, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        if dragging:
            dragging = False
            selected_index = -1
            selected_corner = -1
        elif drawing:
            drawing = False
            fx, fy = x, y
            # Guardar el dibujo
            if draw_mode == 'line':
                drawings.append(('line', (ix, iy), (fx, fy), line_style))
            elif draw_mode == 'circle':
                rx = int(abs(fx - ix))  # Radio horizontal
                ry = int(abs(fy - iy) / 2)  # Radio vertical (mitad del radio horizontal)
                center = ((ix + fx) // 2, (iy + fy) // 2)
                drawings.append(('circle', center, (rx, ry)))
                circles_for_hull.append(center)  # Agregar centro al convex hull
            elif draw_mode == 'rectangle':
                # Ajustar la altura del rectángulo para que sea "acostado"
                width = abs(fx - ix)
                height = int(width / 2)  # Altura proporcional al ancho
                x1 = min(ix, fx)
                y1 = min(iy, fy) - height // 2
                x2 = max(ix, fx)
                y2 = max(iy, fy) + height // 2
                drawings.append(('rectangle', (x1, y1), (x2, y2)))
            elif draw_mode == 'spotlight':
                radius = int(np.sqrt((fx - ix) ** 2 + (fy - iy) ** 2))
                drawings.append(('spotlight', (ix, iy), radius))
            elif draw_mode in ['arrow', 'dashed_arrow']:
                drawings.append(('arrow', (ix, iy), (fx, fy), line_style))
            temp_frame = None  # Limpiar el frame temporal


# Main function to draw on the video
def telestrate_video(video_path):
    global paused, current_frame, drawings, temp_frame, draw_mode, line_style, circles_for_hull, polygon_points

    # Cargar el video
    cap = cv2.VideoCapture(video_path)

    # Verificar si el video se abrió correctamente
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video en la ruta {video_path}")
        return

    # Crear una ventana y asignar la función de callback
    cv2.namedWindow('Telestration')
    cv2.setMouseCallback('Telestration', draw_shape)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            current_frame = frame.copy()  # Guardar el frame actual

        # Dibujar todas las formas almacenadas
        display_frame = current_frame.copy()
        for shape in drawings:
            if shape[0] == 'line':
                if shape[3] == 'solid':
                    cv2.line(display_frame, shape[1], shape[2], (0, 255, 0), 2)
                elif shape[3] == 'dashed':
                    draw_dashed_line(display_frame, shape[1], shape[2], (0, 255, 0), 2)
            elif shape[0] == 'circle' or shape[0] == 'rectangle':
                draw_transparent_shape(display_frame, shape)
            elif shape[0] == 'spotlight':
                display_frame = apply_spotlight(display_frame, shape[1], shape[2])
            elif shape[0] == 'arrow':
                if shape[3] == 'solid':
                    cv2.arrowedLine(display_frame, shape[1], shape[2], (255, 0, 0), 2)
                elif shape[3] == 'dashed':
                    draw_dashed_line(display_frame, shape[1], shape[2], (255, 0, 0), 2)
                    # Dibujar la punta de la flecha manualmente
                    angle = np.arctan2(shape[2][1] - shape[1][1], shape[2][0] - shape[1][0])
                    arrow_tip1 = (int(shape[2][0] - 15 * np.cos(angle + np.pi / 6)),
                                  int(shape[2][1] - 15 * np.sin(angle + np.pi / 6)))
                    arrow_tip2 = (int(shape[2][0] - 15 * np.cos(angle - np.pi / 6)),
                                  int(shape[2][1] - 15 * np.sin(angle - np.pi / 6)))
                    cv2.line(display_frame, shape[2], arrow_tip1, (255, 0, 0), 2)
                    cv2.line(display_frame, shape[2], arrow_tip2, (255, 0, 0), 2)
            elif shape[0] == 'text':
                (tx, ty), text = shape[1], shape[2]
                cv2.putText(display_frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            elif shape[0] == 'triangle':
                draw_inverted_triangle(display_frame, shape[1], shape[2], (255, 255, 255), 0.3)
            elif shape[0] == 'polygon':
                # Dibujar el polígono con relleno transparente y líneas horizontales
                draw_polygon_with_horizontal_lines(display_frame, shape[1], color=(0, 255, 255), alpha=0.3)

        # Dibujar el convex hull si hay suficientes círculos
        if len(circles_for_hull) >= 3:
            draw_convex_hull(display_frame, np.array(circles_for_hull))

        # Mostrar la forma dinámica si se está dibujando
        if temp_frame is not None:
            display_frame = temp_frame.copy()

        # Mostrar el frame
        cv2.imshow('Telestration', display_frame)

        # Esperar la tecla para cambiar el modo de dibujo, eliminar el último elemento o guardar el frame
        key = cv2.waitKey(1) & 0xFF
        if key == ord('l'):  # Cambiar a modo de línea
            draw_mode = 'line'
        elif key == ord('c'):  # Cambiar a modo de círculo
            draw_mode = 'circle'
        elif key == ord('a'):  # Cambiar a modo de flecha
            draw_mode = 'arrow'
        elif key == ord('d'):  # Cambiar a modo de flecha punteada
            draw_mode = 'dashed_arrow'
        elif key == ord('r'):  # Cambiar a modo de rectángulo
            draw_mode = 'rectangle'
        elif key == ord('s'):  # Cambiar a modo de spotlight
            draw_mode = 'spotlight'
        elif key == ord('t'):  # Cambiar a modo de texto
            draw_mode = 'text'
        elif key == ord('v'):  # Cambiar a modo de triángulo
            draw_mode = 'triangle'
        elif key == ord('k'):  # Cambiar a modo de polígono
            draw_mode = 'polygon'
            polygon_points.clear()  # Limpiar la lista de puntos del polígono
            print("Modo polígono activado. Haz clic en 4 puntos del frame.")
        elif key == ord('z'):  # Eliminar el último elemento dibujado
            if drawings:
                drawings.pop()
        elif key == ord('x'):  # Eliminar todos los dibujos
            drawings.clear()  # Limpiar la lista de dibujos
            circles_for_hull.clear()  # Limpiar la lista de círculos para el convex hull
            polygon_points.clear()  # Limpiar la lista de puntos del polígono
            print("Todos los dibujos han sido eliminados.")
        elif key == ord('g'):  # Guardar el frame actual como imagen
            save_frame_with_drawings(current_frame, drawings)
        elif key == ord('h'):  # Limpiar la lista de círculos para el convex hull
            circles_for_hull.clear()
        elif key == ord('p'):  # Pausar/reproducir el video
            paused = not paused
        elif key == ord('m'):  # Cambiar el estilo de línea (sólida/punteada)
            line_style = 'dashed' if line_style == 'solid' else 'solid'
        elif key == ord('q'):  # Salir si se presiona 'q'
            break

    # Liberar el video y cerrar las ventanas
    cap.release()
    cv2.destroyAllWindows()


# Set up the argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Football telestration application.")
    parser.add_argument('video_path', type=str, help="Path to the video file.")
    return parser.parse_args()


# Script entry point
if __name__ == "__main__":
    args = parse_args()
    telestrate_video(args.video_path)
