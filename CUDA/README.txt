Autor:
	Fernando Candelario Herrero

La versión con bordes filtra la imagen completa, se compone de 3 kernels (uno para el noiseReduction, otro para el gradiente,y otro para el hysteresisAndthresholding), de los cuales 2 usan memoria compartida.

La version sin bordes no filtra los bloques limitrofes que componen la imagen el resto de bloques si que son filtrados, se compone de 2 kernels (uno para el noiseReduction, y otro que hace el resto), ambos usan memoria compartida.

En que respecta a tiempos de ejecucion, filtrando la imagen de baseball.bmp ambas versiones están en el mismo intervaloes de tiempos.
