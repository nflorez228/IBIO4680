Cuestionario Lab 1 - Visión por computador 

1) El comando "grep" permite encontrar texto en un archivo

2) Ejecuta un script de tipo Bash

3) Utilizando el comando "getent passwd | grep home" se encuentran el total de usuarios que tienen acceso como usuario al terminal 

   En total son 12 usuarios. 
   
4) 

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
	  
6) wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
tar -xvf BSR_bsds500.tgz

7) du -sh BSR
73M		BSR
find Lab1/BSR/BSDS500/data/images/. | wc -l
507
8)

	```bash
		#!/bin/bash

		# go to Home directory
		cd BSR/BSDS500/data # or just cd


		# find all files whose name end in .tif
		images=$(find images -name *.*)
		for im1 in ${images[*]}
		do
		  #iterate over them
		  identify $im1
		done
      
    ```

9)
10)

	```bash
		#!/bin/bash

		# go to Home directory
		cd BSR/BSDS500/data # or just cd

		# remove the folder created by a previous run from the script
		rm -r cropped_images 2>/dev/null

		# create output directory
		mkdir cropped_images

		# find all files whose name end in .tif
		images=$(find images -name *.*)
		for im1 in ${images[*]}
		do
		  #iterate over them
		  convert -crop 256x256+0+0 $im1 $im1
		done
      
    ```
