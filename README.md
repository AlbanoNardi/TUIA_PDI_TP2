# TUIA_PDI_TP2

Este readme cuenta con descripción detallada de dos scripts de Python diseñados para procesar imágenes y extraer información específica mediante técnicas de procesamiento de imágenes. 
El primer script se utiliza para identificar y clasificar monedas y dados en imágenes, mientras que el segundo se enfoca en la detección y segmentación de caracteres de matrículas.

Script 1: Clasificación de Monedas y Detección de Dados
Descripción
Este script analiza imágenes para identificar monedas y dados. Las monedas se clasifican según su área aproximada, y los dados se analizan para contar los puntos en sus caras visibles.

Requisitos Previos
Python 3.12
Bibliotecas: opencv-python, numpy, matplotlib
Funciones Principales
imshow: Muestra imágenes con opciones de personalización.
rgb_gray_gauss: Convierte una imagen a escala de grises y aplica un filtro gaussiano.
det_circulos: Detecta círculos en la imagen usando la Transformada de Hough.
clasificar_monedas: Clasifica monedas según su área estimada.
det_dados: Detecta los contornos de los dados tras procesar la imagen para eliminar las monedas.
rec_dados: Recorta un dado de la imagen basándose en su contorno.
clasificar_dados: Cuenta los puntos en los dados detectados.
main: Integra todas las funciones para analizar la imagen de entrada.
Uso
Coloca la imagen llamada monedas.jpg en el mismo directorio del script.
Ejecuta el script:
bash
Copiar código
python script2.py
Los resultados se mostrarán en la terminal y en ventanas emergentes con las imágenes procesadas.


Script 2: Detección de Matrículas y Segmentación de Caracteres
Descripción
Este script procesa imágenes para detectar matrículas de vehículos y segmentar los caracteres contenidos en ellas. Utiliza métodos de filtrado de imágenes, detección de contornos y umbralización adaptativa.

Requisitos Previos
Python 3.12
Bibliotecas: opencv-python, numpy, matplotlib
Funciones Principales
imshow: Muestra imágenes con soporte para escala de grises, color y personalización.
patente_umbral: Detecta matrículas en una imagen basándose en características geométricas y de textura.
checkear_tolerancia_consecutiva: Comprueba la uniformidad en las áreas detectadas para validar caracteres.
segmentar_fuerza_bruta: Realiza la segmentación de caracteres de matrículas utilizando diferentes valores de umbral.
main: Procesa un conjunto de imágenes y aplica las funciones anteriores para detectar matrículas y segmentar caracteres.
Uso
Coloca las imágenes en el mismo directorio del script, nombradas como img01.png, img02.png, etc.
Ejecuta el script:
bash
Copiar código
python script1.py
Los resultados se mostrarán en la terminal y en ventanas emergentes con las imágenes procesadas.
