# Laboratorium 01: Wybrane metody przetwarzania sygnałów cyfrowych (a w szczególności obrazów cyfrowych) ( ͡° ͜ʖ ͡°)
## Metody głębokiego uczenia w systemach wizyjnych i wirtualnej rzeczywistości

- Maksymalna liczba punktów: 10
- Skala ocen za punkty:
    - 9-10 ~ bardzo dobry (5.0)
    - 8 ~ plus dobry (4.5)
    - 7 ~ dobry (4.0)
    - 6 ~ plus dostateczny (3.5)
    - 5 ~ dostateczny (3.0)
    - 0-4 ~ niedostateczny (2.0)

Na tym laboratorium zapoznamy się z wybranymi metorami przetwarzania sygnałów, które wykorzystywane są w głębokim uczeniu oraz w wirtuanej rzeczywistości. Użyjemy do tego OpenCV, Keras oraz Unity.

1. [3 punkty] Skalibruj kamerę jednoobiektywową przy użyciu modelu kamery otworkowej (pinhole camera) omówionym na wykładzie. Pozyskaj odpowiednie zdjęcia fotografując kalibrującą szachownicę. Zwróć uwagę, aby szachownica nie miała takiej samej liczby wierszy i kolumn oraz czy linie wykryte w procesie kalibracji są zorientowane w tą samą stronę.

2. [4 punkty] W trójwymiarowym projekcie Unity zbuduj stereowizyjny układ kamer. Następnie utwórz prostą scenę i zapisz rendering z obu kamer do pliku. Oblicz i zwizualizuj mapę dysparycji. Przetestuj różne odległości pomiędzy kamerami oraz wpływ parametrów numDisparities i blockSize na wizualną jakość otrzymanej mapy.

3. [4 punkty] Zaimplementuj przy pomocy biblioteki keras filtry Sobela, Prewitta, Robertsa, filtr Gaussa (o zadanej zmiennymi parametrach sigma oraz wielkości dyskretnego jądra) oraz filtr uśredniający (o zadanej zmiennymi wielkości dyskretnego jądra). Napisz program, który będzie dokonywał tych przekształceń na obrazie z kamery. Do tego rozwiązania wygodnie będzie zastosować API funkcyjne keras zamiast sekwencyjnego. Poniżej znajduje się przykładowa uproszczona implementacja filtru Sobela zrobiona przy pomocy API funkcyjnego:

```py

input = keras.layers.Input(shape=(256,256,1))
la = keras.layers.Conv2D(1, 3, activation=None)(input)
l2 = keras.layers.Conv2D(1, 3, activation=None)(input)
output = keras.layers.Add()([la, l2])
model = keras.Model(inputs=input, outputs=output, name="mnist_model")
model.compile()
model.summary()

# half of the Sobel filter
kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
#kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
kernel = np.transpose(kernel)
kernel = np.expand_dims(kernel, axis=-1)
kernel = np.expand_dims(kernel, axis=-1)
ww = model.layers[1].get_weights()
ww[0] = kernel
model.layers[1].set_weights(ww)

# second half of the Sobel filter
kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
kernel = np.transpose(kernel)
#kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
kernel = np.transpose(kernel)
kernel = np.expand_dims(kernel, axis=-1)
kernel = np.expand_dims(kernel, axis=-1)
ww = model.layers[2].get_weights()
ww[0] = kernel
model.layers[2].set_weights(ww)

```

a to implementacja przy użyciu zdefiniowanej warstwy obliczającej normę z gradientu:

```py

from keras import layers
from keras import backend as K

class TensorsSumNormNorm(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        res = inputs[0] * inputs[0] + inputs[1] * inputs[1]
        return K.sqrt(res)


input = keras.layers.Input(shape=(256,256,1))
la = keras.layers.Conv2D(1, 3, activation=None)(input)
l2 = keras.layers.Conv2D(1, 3, activation=None)(input)
output = TensorsSumNormNorm()([la, l2])
model = keras.Model(inputs=input, outputs=output, name="mnist_model")


model.compile()
model.summary()

```