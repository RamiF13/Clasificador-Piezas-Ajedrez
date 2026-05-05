import numpy as np
import tensorflow as tf
import cv2

modelo = tf.keras.models.load_model("modelo_ajedrez.keras")
camara = cv2.VideoCapture(0)
contador = 0
nombres = ['Alfil blanco', 'Alfil negro', 'Caballo blanco', 'Caballo negro', 'Dama blanca', 'Dama negra', 'Peon blanco', 'Peon negro', 'Rey blanco', 'Rey negro', 'Torre blanca', 'Torre negra']
nombre = ""
confianza = 0.0

while True: 
    ret, frame = camara.read()
    if ret == False:
        break
    contador += 1
    if contador % 20 == 0:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(224,224))
        img = np.array(img)
        img = img * 1/255
        img = np.expand_dims(img,axis=(0))
        prediccion = modelo.predict(img)
        clase = np.argmax(prediccion)
        nombre = nombres[clase]
        confianza = prediccion[0][clase]
    cv2.putText(img=frame,org=(10,30),thickness=2,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1, text=nombre+" "+str(round(confianza,2)),color=(0,255,0))
    cv2.imshow("Que pieza es?",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
camara.release()
cv2.destroyAllWindows()
    