# A basic neural network with examples
## Caratteristiche:
La rete neurale si propone di:
   1. dividere opportunatamente in training e test set con un rapporto approssimato a 1:4
   2. fissare la discesa del gradiente con il momento
   3. studiare l'andamento della rete per derivare:
      - numero epoche necessarie all'apprendimento
      - andamento errore su training e validation set, accuratezza su test set con un singolo strato al variare del learning rate
         e dal momento per almeno 5 diverse dimensioni dello strato interno
      - lasciare invariati parametri come funzioni di output

## Local installation
To run the project on your operating system make sure you already have python 3 installed and then install the following packages:
```
   RUN pip install --upgrade pip
   RUN pip install opencv-python-headless
   RUN pip install matplotlib keras tensorflow
   RUN pip install pandas
```
## Docker
If you have a docker environment run the build phase with the command ```docker build -t nndl .``` In the project folder once.<br>
To test the changes or to run the neural network run:
-  ```docker container run --rm -v .:/app nndl ``` for visualize matplotlib's graphics
-   ```docker container run --rm -v .:/app nndl ``` for visualize only string result
