# A basic neural network with examples
## Local installation
To run the project on your operating system make sure you already have python 3 installed and then install the following packages:
```
   RUN pip install --upgrade pip
   RUN pip install opencv-python-headless
   RUN pip install matplotlib keras tensorflow
```
## Docker
If you have a docker environment run the build phase with the command ```docker build -t nndl .``` In the project folder once.<br>
To test the changes or to run the neural network run:
-  ```docker container run --rm -v .:/app nndl ``` for visualize matplotlib's graphics
-   ```docker container run --rm -v .:/app nndl ``` for visualize only string result
