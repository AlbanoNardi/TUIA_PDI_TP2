import cv2
import numpy as np
import matplotlib.pyplot as plt

# Defininimos función para mostrar imágenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)

def patente_canny(imagen_path):
        """
        Detecta la patente utilizando Canny y criterios de contornos.
        """
        # Leer imagen y convertir a escala de grises
        imagen = cv2.imread(imagen_path)
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        # Aplicar filtro Gaussiano y Canny
        f_blur = cv2.GaussianBlur(gray, ksize=(3, 3), sigmaX=1.5)
        edges = cv2.Canny(f_blur, threshold1=102, threshold2=190)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for contorno in contours:
            x, y, w, h = cv2.boundingRect(contorno)
            aspect_ratio = w / h
            area = cv2.contourArea(contorno)
            rect_area = w * h
            fill_ratio = area / rect_area

            # Filtrar contornos que cumplan los criterios de patente
            if 1.0 <= aspect_ratio <= 3.5 and 0.5 < fill_ratio < 0.9 and 2000 < rect_area < 3000:
                # Recortar y mostrar la patente detectada
                patente_recortada = imagen[y:y+h, x:x+w]
                imshow(patente_recortada, title="Patente Detectada (Canny)")
                return patente_recortada
        return None

def reconocer_patente(imagen_path):
        """
        Detecta la patente utilizando umbral adaptativo y filtros de contorno.
        """
        # Leer imagen y convertir a escala de grises
        imagen = cv2.imread(imagen_path)
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        # Filtro pasa-altos
        kernel = -np.ones((5, 5)) / (5 * 5)
        kernel[2, 2] = 24 / 25
        img_filtered = cv2.filter2D(gray, -1, kernel)
        
        # Aplicar umbral adaptativo
        thresh = cv2.adaptiveThreshold(img_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contorno in contours:
            x, y, w, h = cv2.boundingRect(contorno)
            aspect_ratio = w / h
            area = cv2.contourArea(contorno)
            rect_area = w * h
            fill_ratio = area / rect_area

            # Filtrar contornos que cumplan los criterios de patente
            if 1.0 <= aspect_ratio <= 3.5 and 0.5 < fill_ratio < 0.9 and 900 < rect_area < 3800:
                patente_recortada = imagen[y:y+h, x:x+w]
                imshow(patente_recortada, title="Patente Detectada (Método Alternativo)")
                return patente_recortada
        return None

# 1. Detección de la patente
def detectar_patente(imagen_path):
    """
    Detecta la patente en la imagen utilizando dos métodos complementarios: Canny y un método alternativo.
    """

    # Intentar detección con ambos métodos
    patente = patente_canny(imagen_path)
    if patente is not None:
        print("Patente detectada con Canny.")
        return patente
    
    print("Intentando método alternativo...")
    patente = reconocer_patente(imagen_path)
    if patente is not None:
        print("Patente detectada con método alternativo.")
        return patente

    print("No se encontró una patente adecuada con ninguno de los métodos.")
    return None

# 2. Segmentación de caracteres
def segmentar_caracteres(patente):
    """
    Segmenta los caracteres de la patente detectada.
    """
    # Convertir a escala de grises y ecualizar
    gray = cv2.cvtColor(patente, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    
    # Umbralización
    _, thresh_img = cv2.threshold(gray_eq, thresh=118, maxval=255, type=cv2.THRESH_BINARY)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    caracteres = []
    for contorno in contours:
        x, y, w, h = cv2.boundingRect(contorno)
        aspect_ratio = w / h
        area = cv2.contourArea(contorno)

        # Filtrar contornos que parezcan caracteres
        if 30 < area < 90 and 0.3 < aspect_ratio < 0.7:
            cv2.rectangle(patente, (x, y), (x + w, y + h), (0, 255, 0), 2)
            caracter_recortado = gray[y:y+h, x:x+w]
            caracteres.append((x, caracter_recortado))
    
    # Ordenar caracteres por su posición horizontal
    caracteres = sorted(caracteres, key=lambda c: c[0])

    # Mostrar la patente con caracteres resaltados
    imshow(patente, title="Caracteres Segmentados")
    return caracteres

# 3. Flujo principal
def main(imagen_path):
    """
    Flujo principal para detectar y segmentar caracteres de una patente.
    """
    patente = detectar_patente(imagen_path)
    if patente is not None:
        caracteres = segmentar_caracteres(patente)
        print(f"Número de caracteres detectados: {len(caracteres)}")
        return caracteres
    else:
        print("No se pudo procesar la imagen.")
        return None

# Ejecutar sobre una imagen
caracteres_detectados = main('img01.png')
caracteres_detectados = main('img02.png')
caracteres_detectados = main('img03.png')
caracteres_detectados = main('img04.png')
caracteres_detectados = main('img05.png')
caracteres_detectados = main('img06.png')
caracteres_detectados = main('img07.png')
caracteres_detectados = main('img08.png')
caracteres_detectados = main('img09.png')
caracteres_detectados = main('img10.png')
caracteres_detectados = main('img11.png')
caracteres_detectados = main('img12.png')