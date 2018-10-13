'''
Autor: Anderson Alves Schinaid
Data: 20/09/2018
@Base no curso python DeepLearning de A - Z
@livro Simon Haykin Redes neurais Principios e pratica 
'''


from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import keras
from tkinter import *
import cv2
import numpy as np
#from tkinter import Image
from PIL import Image, ImageTk
import time
#from tkinter import ImageTk
#from keras.preprocessing.image import ImageDataGenerator

# dimensions of our images
    
def bt_verificar():
    
    img_width, img_height = 256, 256
# load the model we saved
    classifier = load_model('model/model_black.h5')
    classifier.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

# predicting images
    img = image.load_img('data/predicao_img/tb 31.jpg', target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = classifier.predict_classes(images, batch_size=10)
    print(classes)

# predicting multiple images at once
    img = image.load_img('data/predicao_img/tb 31.jpg', target_size=(img_width, img_height))
    y = image.img_to_array(img)
    y = np.expand_dims(y, axis=0)

# pass the list of multiple images np.vstack()
    images = np.vstack([x, y])
    classes = classifier.predict_classes(images, batch_size=10)
#classifier.class_indices
    if classes[1]:
        classes = 'Plantação de tomate com bacteria'
 #   elif classes[0]:
  #      classes = 'Plantação do tomate com Fungo de Alternaria solani'
   # elif classes[3]:
    #   classes = 'Plantação do tomate Late Blight'
    #elif classes[4]:
     #   classes = 'Plantação do tomate com Septoria left spot'
    else:
        classes = 'problema não catalogado'
# print the classes, the images belong to
    print (classes)
#print (classes[0])
#print (classes[0][0])
    lb ["text"] = classes

#iniciando a janela do projeto 
janela = Tk()
#iniciando o frame do mesmo 
imageFrame = Frame(janela, width=100, height=100)

#tirando foto usando o opencv 
def tirar_foto():
    emLoop = True
    while(emLoop):
        cv2.waitKey(30)&0xff == ord('y')
        cv2.imwrite('data/Save_Image/teste1.jpg', frame)
        cv2.destroyAllWindows()
        break
    cap.release()
def ligar_camera():
    emLoop = True
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    nFrames = 30
    while(emLoop):
        ret, frame = cap.read()
        #frame= cv2.QueryFrame(cap)
        cv2.imshow('frame', frame)
        cv2.imwrite('data/Save_Image/teste.jpg', frame)
        k = cv2.waitKey(100)
        if k == 27:
            emLoop = False
        if k == ord('s'):
            cv2.imwrite('data/Save_Image/teste1.jpg', frame)
            emLoop = False
    cap.release()
    cv2.destroyAllWindows()

def bt_sair():
    janela.destroy()

#construção padrão do botão e label e suas configurações de tamanho comando e texto 
bt_verificar = Button(janela, width =20, text="verificar", command=bt_verificar)
bt_verificar.place(x=10, y=250)

bt_ligar_camera = Button(janela, width=20, text="Ligar camera", command=ligar_camera)
bt_ligar_camera.place(x=170, y=250)

bt_tirar_foto = Button(janela, width =20, text="Tirar foto", command=tirar_foto)
bt_tirar_foto.place(x=330, y=250)
#capture_image()

lb = Label(janela, text="pressione o botão verificar para analisar a imagem" )
lb.place(x=10, y=10)
janela.title("Deep Learning")
janela.geometry("500x300+300+200")
janela.mainloop()
