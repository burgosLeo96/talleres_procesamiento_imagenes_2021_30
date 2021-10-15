# Taller 5
## Leonard Stuart Burgos Jiménez

## Cómo usar

- Clonar el repositorio
- Dentro de la carpeta "taller 6" del repositorio, ejecutar la siguiente sentencia por línea de comandos:
```sh
python main.py <directorio_donde_se_encuentran_las_imágenes>
```

- El directorio recibido debe contener una lista de imágenes nombradas de la manera "image_(número).jpg" de manera
  en orden en el que fueron tomadas las fotos. A continuación, se debe suministrar al programa un número entre 1 y la
  cantidad de imágenes para seleccionar la imagen de referencia. Posteriormente, se debe seleccionar la estrategia a usar
  para la identificación de puntos de interés (SIFT o ORB. Tener en cuenta las mayúsculas) El programa se encargará de 
  identificar los puntos de interés entre cada par de imágenes para así crear la homografía con los mejores puntos. 
  La imagen resultado será mostrada en pantalla y se almacenará bajo el nombre "result_image.jpg"