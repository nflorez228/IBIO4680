Cuestionario Lab 1 - Visión por computador 

1) El comando "grep" permite encontrar texto en un archivo

2) Ejecuta un script de tipo Bash

3) Utilizando el comando "getent passwd | grep home" se encuentran el total de usuarios que tienen acceso como usuario al terminal 

   En total son 12 usuarios. 
   
4) 

5) 
	```bash
      #!/bin/bash
      
      # go to Home directory
      cd ~ # or just cd

      # remove the folder created by a previous run from the script
      rm -r color_images 2>/dev/null

      # create output directory
      mkdir color_images

      # find all files whose name end in .tif
      images=$(find sipi_images -name *.tiff)
      
      #iterate over them
      for im in ${images[*]}
      do
         # check if the output from identify contains the word "gray"
         identify $im | grep -q -i gray
         
         # $? gives the exit code of the last command, in this case grep, it will be zero if a match was found
         if [ $? -eq 0 ]
         then
            echo $im is gray
         else
            echo $im is color
            cp $im color_images
         fi
      done
      
      ```