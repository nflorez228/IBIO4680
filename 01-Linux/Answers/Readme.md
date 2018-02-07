Cuestionario Lab 1 - Visión por computador 

1) El comando "grep" permite encontrar texto en un archivo

2) Ejecuta un script de tipo Bash

3) Utilizando el comando "getent passwd | grep home" se encuentran el total de usuarios que tienen acceso como usuario al terminal 

   En total son 12 usuarios. 
   
4) El comando para tener una tabla de usuarios es: 

  ``getent passwd | grep home | cut -d ":" -f 1``

5) 
	```bash
		#!/bin/bash

		# remove the folder created by a previous run from the script
		rm -r duplicate_images 2>/dev/null

		# create output directory
		mkdir duplicate_images

		# find all files whose name end in .tif
		images=$(find sipi_images -name *.tiff)
		for im1 in ${images[*]}
		do
		  #iterate over them
		  for im in ${images[*]}
		  do
		  #echo "$im"
			if [ $im == $im1 ]
			then
			 echo ""
			else
			   md5sum < $im1 | cut -d\  -f1 > md5sum1.txt
			   md5sum < $im | cut -d\  -f1 > md5sum.txt
			   if cmp --silent md5sum1.txt md5sum.txt
			   then
				  echo $im1 and $im are duplicates
			   
				  #echo $im1 and $im arent duplicates
				  cp $im duplicate_images
			   fi
			fi
		  done
		done
      
      ```
	  
6 Para descargar el archivo usamos el comando ``wget`` seguido del link de descarga.
``wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz``
Una vez descargado se descomprime el archivo con el comando ```tar`` asi:
``tar -xvf BSR_bsds500.tgz``

7) El comando ``du -sh`` nos permite ver el tamaño de una carpeta con todos sus archivos recursivos
``du -sh BSR``
El comando nos resulta:
``73M		BSR``
Lo cual significa que la carpeta BSR pesa 73M

Para saber cuantos elementos hay en una carpeta se utiliza el comando ``find`` en conjunto con el comando de conteo ``wc -l`` asi:
``find Lab1/BSR/BSDS500/data/images/. | wc -l``
la respuesta son: ``507`` archivos

8) Se creo un programa que recorre las imagenes bajo ``BSR/BSDS500/data`` y a este usa el comando ``identify`` para saber su resolucion y su formato.
	```bash
		#!/bin/bash

		# go to Home directory
		cd BSR/BSDS500/data # or just cd


		# find all files
		images=$(find images -name *.*)
		for im1 in ${images[*]}
		do
		  #iterate over them
		  identify $im1
		done
      
    ```

9) Para saber la orientacion de cada imagen bajo la carpeta ``BSR/BSDS500/data`` se recorren las imagenes y se cuentan cauntas cumplen el criterio de proporcion de ancho/alto.
Esto se logra obteniendo las proporciones con el comando ``convert $im1 -format "%[fx:(w/h>1)?1:0]" info:`` y luego se pregunta si es 1 o 0, Landscape o Portrait.
	```bash
		#!/bin/bash

		# go to Home directory
		cd BSR/BSDS500/data # or just cd


		# find all files
		images=$(find images -name *.*)
		count=0
		for im1 in ${images[*]}
		do
		#iterate over them
		value=$(convert $im1 -format "%[fx:(w/h>1)?1:0]" info:)
		if [ $value -eq 1 ]
		then
		((count++))
		fi
		done
		echo "hay" $count "Landscape"
      
    ```
	La respuesta del programa son 348 imagenes.
	
10) Para cortar las imagenes se itera sobre la carpeta ``BSR/BSDS500/data`` y a cada imagen se utiliza el comando ``convert`` con el subcomando ``-crop`` especificando las dimensiones deseadas (256x256)
	```bash
		#!/bin/bash

		# go to Home directory
		cd BSR/BSDS500/data # or just cd

		# remove the folder created by a previous run from the script
		rm -r cropped_images 2>/dev/null

		# create output directory
		mkdir cropped_images

		# find all files
		images=$(find images -name *.*)
		for im1 in ${images[*]}
		do
		  #iterate over them
		  convert -crop 256x256+0+0 $im1 $im1
		done
      
    ```
