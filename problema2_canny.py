import cv2
import numpy as np
import matplotlib.pyplot as plt
import canny 

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

# canny: 102, 191.25
# canny

def patente_canny(imagen):       # encuentra el 1
    # leo la imagen, la paso a escala de grises y le aplico blur
    imagen = cv2.imread(imagen)
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)  # Pasamos a escala de grises
    f_blur = cv2.GaussianBlur(gray, ksize=(3, 3), sigmaX=1.5) # o sigma 2
    #img_canny_CV2 = cv2.Canny(img_blur, 150, 255, apertureSize=3, L2gradient=True) 
    gcan3 = cv2.Canny(f_blur, threshold1=102, threshold2=190)  # o 50 115
    imshow(gcan3)
    #contours, hierarchy = cv2.findContours(img_canny_CV2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  
    contours, hierarchy = cv2.findContours(gcan3, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  
    # contours_area = sorted(contours, key=cv2.contourArea, reverse=True)
    # cv2.drawContours(imagen, contours_area, contourIdx=1, color=(0, 255, 0), thickness=2) # este encuentra la patente
    # cv2.imshow('Contorno ordenados por area', imagen)
    for contorno in contours:
            # Obtener el rectángulo delimitador
            x, y, w, h = cv2.boundingRect(contorno)
            aspect_ratio = w / h  # Relación de aspecto
            area = cv2.contourArea(contorno)
            rect_area = w * h
            fill_ratio = area / rect_area  # Proporción de área cubierta por el contorno
            # Filtrar por criterios típicos de patentes
            if 1.0 <= aspect_ratio <= 3.5 and 0.9 > fill_ratio > 0.5 and 3000 > rect_area > 2000:
                # Dibujar el contorno detectado
                cv2.drawContours(imagen, [contorno], -1, (0, 255, 0), 2)
                patente_recortada = imagen[y:y+h, x:x+w]
                cv2.imshow("Patente Detectada", patente_recortada)
                cv2.imshow("Contorno en Imagen", imagen)
                return patente_recortada
    print("No se encontró una patente adecuada.")
    return None

patente_canny('img03.png')    


# dif = img_canny_CV2.astype(np.int16) - img_canny_CV2.astype(np.int16)

# esta funcion para canny agarra img_blur, 150, 255, apertureSize=3, L2gradient=True
# este ademas aplica canny.canny(img_blur, th1=50, th2=115)
