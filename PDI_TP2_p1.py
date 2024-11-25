import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
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

def rgb_gray_gauss(img):
    img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imshow(img_color, title="Imagen en RGB", color_img=True)

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imshow(img_gray, title="Imagen en escala de grises")

    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    plt.figure()
    ax1 = plt.subplot(121)
    imshow(img_gray, title='Imagen original en escala de grises', new_fig=False)
    plt.subplot(122, sharex=ax1, sharey=ax1)
    imshow(img_blur, title='+ Filtro Gaussiano', new_fig=False)
    plt.show(block=False)
    return img_blur

def det_circulos(img_procesada,img):
    circles = cv2.HoughCircles(img_procesada, cv2.HOUGH_GRADIENT, 1, minDist=200, param1=150, param2=40, minRadius=100, maxRadius=200)

    img_color_ = img.copy()
    circles = np.uint16(np.around(circles))  # Redondear coordenadas
    for i in circles[0, :]:
        cv2.circle(img_color_, (i[0], i[1]), i[2], (0, 255, 0), 3)  # Círculo externo
        cv2.circle(img_color_, (i[0], i[1]), 2, (0, 0, 255), 3)  # Centro
    imshow(img_color_, title="Círculos detectados")
    return circles

def clasificar_monedas(circles):
    monedas = {"1 peso": 0, "50 centavos": 0, "10 centavos": 0} # Diccionario de monedas
    for (x, y, radio) in circles[0]:
        area = math.pi * radio**2
        if 70000 <= area < 95000:
            monedas["1 peso"] += 1
        elif area > 95000:
            monedas["50 centavos"] += 1
        elif area < 70000:
            monedas["10 centavos"] += 1
    print("Clasificación de monedas:", monedas)


def det_dados(circles,img_procesada):
    
    for i in circles[0, :]:
        cv2.circle(img_procesada, (i[0], i[1]), i[2] + 10, (0, 0, 0), thickness=cv2.FILLED)  # Tapamos las monedas para detectar los dados
    imshow(img_procesada, title="Monedas tapadas")

    _, thresh_img = cv2.threshold(img_procesada, thresh=180, maxval=255, type=cv2.THRESH_BINARY)
    imshow(thresh_img, title="Imagen umbralizada")

    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 500:  # Umbral para limpiar areas chicas
            cv2.drawContours(thresh_img, [cnt], -1, (0, 0, 0), thickness=cv2.FILLED)
    imshow(thresh_img, title="Filtrado de areas pequeñas")

    A = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    C = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    img_ap = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, A)
    img_cl = cv2.morphologyEx(img_ap, cv2.MORPH_CLOSE, C)

    plt.figure()
    ax1 = plt.subplot(131)
    imshow(thresh_img, new_fig=False, title="Original")
    plt.subplot(132, sharex=ax1, sharey=ax1)
    imshow(img_ap, new_fig=False, title="Apertura")
    plt.subplot(133, sharex=ax1, sharey=ax1)
    imshow(img_cl, new_fig=False, title="Apertura + Clausura")
    plt.show(block=False)

    
    contours, _ = cv2.findContours(img_cl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Obtenemos los contornos de los dados
    
    return contours,img_cl

def rec_dados(contorno, img):
    img_inv = img==0
    inv_uint8 = img_inv.astype(np.uint8)
    x, y, w, h = cv2.boundingRect(contorno)
    return inv_uint8[y:y+h, x:x+w]

def clasificar_dados(contours,img_cl):
    for idx, contorno in enumerate(contours):
        dado = rec_dados(contorno, img_cl)
        imshow(dado)
        cont_int, _ = cv2.findContours(dado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        numero = 0
        for cnt in cont_int:
            if 1500 < cv2.contourArea(cnt) < 3500:
                numero += 1
        print(f"Dado {idx}: {numero} puntos detectados")


def main():

    imagen_path='monedas.jpg' # Path de la imagen a procesar
    
    img = cv2.imread(imagen_path) # Lectura de la imagen
    
    imshow(img, title="Imagen Original") # Carga y muestra de la imagen original
    
    img_procesada = rgb_gray_gauss(img) # Preprocesamiento, escala de grises + filtro gaussiano
    
    circulos_monedas = det_circulos(img_procesada,img) # Detección de bordes de las monedas
    
    contorno_dados,img_cl = det_dados(circulos_monedas,img_procesada) # Deteccion de dados de la imagen
    
    clasificar_dados(contorno_dados,img_cl) # Clasificación de los dados
    
    clasificar_monedas(circulos_monedas) # Clasificación de las monedas


if __name__ == "__main__":
    main()