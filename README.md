# frp_learning_datathon


RETO MINSAIT REAL STATE MODELLING
FRP Learning 
Aitor Mira Abad
Fermín Leal Payá
Fecha: 2019-03-15

# 1. TRATAMIENTO DE IMÁGENES
# 1.1 Carga y Extracción de Datos de Imágenes
El primer paso a dar en el presente reto es el procesado de imágenes. Con el objetivo de tratarlas más adelante junto a los datos de entrenamiento, vamos a obtener la siguiente información:
1. Nombre
2. Altura
3. Anchura
4. Histograma en rgb
5. Histograma en greyscale
6. Imagen en rgb (flat array)
7. Imagen en grayscale (flat array)
8. Numero de imágenes por inmueble

En el siguiente paso se cargarán todas las imágenes y crear una lista de diccionarios que contengan la información que queremos extraer. Las imágenes serán transformadas a 50x50 para poder tratarlas más rápidamente, ya que si no el tiempo de cómputo sería inasumible. Otra de las tareas a realizar es limpiar el nombre de las imágenes, para conservar simplemente el identificador de las mismas y así puedan ser cruzadas con los datos posteriormente. Una vez extraída la información y el diccionario creado, se almacenará con la librería pickle de serialización de archivos.
La lista de diccionarios generada se debe transformar a un formato que permita cruzarlo con los datos. Se ha elegido un DataFrame Pandas como estándar en tratamiento de datos. Para esto se creará una matriz que será transformada a pd.DataFrame y lo será almacenada en un CSV con el nombre de archivo 'df_imagenes.csv'.
Posteriormente se ejecuta un código que tiene como finalidad contar las imágenes correspondientes a cada observación. De esta se conocerá cuantas imágenes tiene cada anuncio analizado, lo que puede aportar gran cantidad de información al modelo. Un usuario que entra con intención de comprar un inmueble, evidentemente requiere ver fotos del mismo, y la cantidad de fotos puede ser decisiva para determinar el tiempo que pase en la página del anuncio.
1.2 Reducción de Dimensión
Se procede conversión final para poder juntar los datos con el dataset de entrenamiento de forma eficiente mediante la técnica de reducción dimensional PCA.

# 2. TRATAMIENTO DE DATOS
# 2.1 Carga de Datos
En este apartado procedemos con la limpieza y procesamiento del dataset "Modelar_UH2019.txt" para poder entrenar el modelo, así como el "Estimar_UH2019.txt" para su predicción.
El tratamiento de las imágenes ha generado una gran cantidad de información a utilizar para el problema de regresión. Se ha utilizado un proceso lógico de prueba-error iterativo en el que se ha comprobado la cantidad y calidad de información que aportaban todas estas variables al modelo analizando los distintos outputs en que ha resultado. Tras su estudio en detalle se ha encontrado una relación clara con la información de las siguientes columnas:
1. Name
2. Height
3. With
4. Count
Estas columnas, que se corresponden con los metadatos de las imágenes, han conseguido mejorar considerablemente la regresión del modelo mientras que el resto de datos, referentes al contenido de las imágenes, no han arrojado ningún tipo de mejora sobre el modelo.

Para verificar que el contenido de las imágenes realmente no estuviera aportando ninguna mejora al modelo, se ha preparado una clusterización de imágenes mediante un modelo K-MEANS con un número de clústers óptimo. Esta agrupación de imágenes por su contenido habría aportado información en el caso de existir una fuerte relación, en cambio esta metodología tampoco ha conseguido mejorar la regresión.
# 2.2 Features Engineering
En este apartado se tratarán los datos de forma oportuna según los requisitos de entrada de los modelos de análisis y de forma que aporten la mayor información posible al propio modelo.
En primer lugar, se tratarán las columnas cuya información es de tipo categórico. Estas columnas tienen valores numéricos o de texto que requerirán un tratamiento para que el modelo pueda interpretar la información. Para las columnas caracterizadas con valores únicos, se ha realizado una transformación de esos valores en códigos. Está transformación permite obtener en una misma columna un código numérico para cada una de las categorías de la columna.
Se ha optado por esta aproximación al problema ya que el tratamiento categórico, con una posterior transformación en columnas dummies afectaba negativamente al modelo obteniendo un error mayor.
Las variables de texto libre se tratarán posteriormente.
# 2.3 Natural Languaje Processing (NLP)
A continuación, se tratarán las columnas con información en formato de texto abierto. Estas columnas de texto descriptivo pueden aportar gran cantidad de información por lo que se ha prestado especial atención:

HY_descripcion
HY_distribucion
El análisis de estas features se ha abordado basandose en la téncia de "bag of words", que consiste en análizar las palabras existentes para convertirlas en features de forma que cada observación tendrá un valor que se corresponde con el número de veces que aparece esa palabra en el texto (descripción o distribución) de esa observación. Haciendo uso de los métodos de tf-idf o CountVectorizer se procede a la creación de la "bag of words", añadiendo además interacciones de palabras mediante los llamadas "n-grams" que agrupa palabras consecutivas posibilitando al modelo captar la información de estructuras de varias palabras que se suelen utilizar juntas.

Esta aproximación resulta demasiado simplista y abusa de "fuerza bruta" resultando en modelos pesados con mucha información, aunque con una gran cantidad de ruido que impide la mejora del modelo. A consecuencia de esto se procede a emplear técnicas de selección de features.

Se procede con la selección de palabras o combinación de palabras (n-grams) que ofrezcan mayor correlación con el TARGET. Para ello se utilizan técnicas tipo SelectKBest o análisis más básicos como la frecuencia total de aparición de las palabras en el texto. Con esto, se ejecuta una primera aproximación a la mejor solución, pues ya se obtiene un grado de mejora sustancial en el modelo, sin embargo, no resulta suficientemente significativa.

Llegados a este punto, se opta por intentar ejecutar este mismo análisis desde una perspectiva más simple y manual. En conjunto con las palabras obtenidas con los métodos anteriores, se optó por elaborar una lista de palabras clave que más llaman la atención de las personas que ven anuncios y trabajar con ellas. Para ello, se ha utilizado información obtenida de las propias páginas que ofertan inmuebles, donde se indican las palabras que habitualmente dan buenos resultados al incluirlas en la descripción.
https://www.idealista.com/news/inmobiliario/vivienda/2017/11/24/749048-consejos-practicos-para-conseguir-una-buena-descripcion-de-los-inmuebles
Esta aproximación final, que aúna el proceso de selección de todas las anteriores junto con la experiencia de los propios analistas en base a información real, resulta en el mejor resultado del modelo obtenido mediante NLP.
El procedimiento seguido hasta la aproximación final queda descrito en el ANEXO I. NLP.
Adicionalmente, se han creado features con la longitud de ambos textos.
# 2.4 Feature Interactions
Con el objetivo de mejorar la regresión se han generado múltiples interacciones entre columnas con el objetivo de alimentar el modelo con información lógica obtenida de relaciones subyacentes a los datos. Las relaciones que se han probado son:
1. Tiempo del anuncio on-line (en quincenas)
2. Promedio de visitas diarias
3. Variación porcentual de precio
4. Variación absoluta de precio
Aunque estas relaciones resultan triviales, solo la variación absoluta de precio y la duración del anuncio han mejorado el modelo.
# 2.5 TARGET Engineering
# 2.5.1 Linealización del TARGET
Con el objetivo de comprender la variable a objetivo a predecir, se procede a estudiar la distribución del TARGET. Para ello, se va a caracterizar con los parámetros descriptores de la forma de la distribución de la variable:
1. Skewness. Se trata de una medida de asimetría de la distribución respecto de la media de la variable.
2. Kurtosis. Al igual que el Skewness, mide la forma de las colas de la distribución.
Se obtienen unos valores positivos muy altos de Skewness y Kurtosis, lo que indica que la distribución de la variable presentará una importante asimetría hacia la derecha de la media. Resulta evidente comprobar estos resultados gráficamente.
Probada la asimetría de la distribución, su larga cola hacia la derecha y la lógica acumulación de valores en torno a 0, se procede a aplicar el logaritmo de (target+1) sobre la variable para corregir esta asimetría.
La elección de esta transformación logarítmica se basa en su efectividad al regularizar valores elevados. La utilización del logaritmo de TARGET + 1 se entiende por el gran número de valores 0 para los que el logaritmo tiende asintóticamente a menos infinito.
Resulta evidente la normalización de la distribución que implicará una mejora del modelo. Para cuantificar la normalización, se vuelven a calcular los estimadores de asimetría de la distribución y se aplica definitivamente el logaritmo a TARGET.
Skewness: -1.099007 Kurtosis: 2.111987
Tanto la Kurtosis como el Skewness se han reducido drásticamente presentando valores de desviación de la media aceptables.
Alternativamente a la normalización logarítmica, se ha trabajado con una transformación de tipo Box-Cox optimizada para el problema, sin embargo, sus resultados han sido ligeramente peores (décimas de diferencia) a los obtenidos por la transformación logarítmica.
# 2.5.2 TARGET Anómalo¶
Del análisis del TARGET y posterior normalización se han desprendido conclusiones acerca de la normalidad de la distribución y su idoneidad para el modelo. Sin embargo, existe un detalle una gran acumulación de valores en torno a 0 que requiere de un estudio en profundidad.
En primer lugar, cabe señalar que estos valores carecen de sentido físico ya que las visitas de 0 segundos son, por definición, anómalas. Por ello, se inspecciona visualmente la variable TARGET para tratar de identificar patrones en estos valores.
Resulta evidente que cuando las visitas a la página son 0, o que cuando la tasa de salida es del 100 %, el valor de TARGET siempre es 0. Los valores observados como 25.32 se corresponden con los valores dummy de los registros a estimar, los cuales se predecirán automáticamente como 0.
Con estos criterios, se han identificado un total de 29 valores del dataset a estimar cuyos valores serán 0 y se podrán excluir de la predicción. Asimismo, los valores 0 del dataset a modelizar podrán ser excluidos del entrenamiento del modelo.
# 2.5 Feature Correlation
Con el objetivo de evaluar la validez de las features disponibles, sobre todo de aquellas creadas en el proceso de Feature Engineering, se procede a analizar las correlaciones de cada una con la variable TARGET y eliminar aquellas que no aporten información, es decir, cuya correlación sea nula.
# 3. Modelización
En este apartado se preparan los datos para su entrenamiento y evaluación. Posteriormente se seleccionará el modelo óptimo para el problema. Finalmente se entrenará el modelo y se ejecutará sobre el dataset a estimar.
# 3.1 Preparación de Datos
Se procede a preparar los datasets de modelado para su introducción al modelo. De esta forma se separarán definitivamente los sets de modelización y predicción. Dado que el modelo finalmente empleado trabaja con todo el dataset de modelización mediante cross-validation, no resulta necesario separar en sets de entrenamiento y validación, sin embargo, esta separación ha sido utilizada ampliamente en el proceso previo a alcanzar este modelo.
Como resultado de todas las iteraciones realizadas en los distintos modelos se optimizó el dataset utilizado para el entrenamiento filtrando las observaciones con valores de TARGET superiores al percentil 0.766. Este proceso se explica detalladamente en el ANEXO I.
Por otro lado, se han eliminado los registros con TARGET = 0, como se ha explicado previamente, para ser posteriormente añadidos al resultado final.
# 3.2 Definición de Modelo
Este apartado a concentrado gran parte de los esfuerzos de este proyecto, sobre todo en la última semana. Este trabajo queda reflejado en el ANEXO I donde se explica cada una de las aproximaciones empleadas a lo largo del proyecto y el proceso hasta alcanzar el modelo de stacking definido.
# 3.3 Entrenamiento y Validación
Para evaluar la validez del modelo de stacking escogido se ha utilizado un modelo de cross-validation aplicado al stacking que permitirá evaluar su poder de predicción y generalización. Cabe destacar que el MAE obtenido en esta ejecución será sensiblemente inferior a la real debido a que se han retirado de dataset los registros por encima del percentil 0.766 tal y como se ha explicado en 3.1.
# 4. Predicción y entrega
Obtenido el mejor valor de evaluación mediante el stacking, se procede al entrenamiento y predicción del dataset completo para su entrega definitiva.
ANEXO I. OPTIMIZACIÓN
AI.1 Búsqueda de parámetros y modelos óptimos
En el tiempo trascurrido desde el planteamiento de este problema, se ha descubierto que no es, ni mucho menos, trivial. Por este motivo se han probado multitud de regresores provistos por SciKit Learn y Keras (TensorFlow background). Además de la prueba de los modelos con parámetros por defecto, se ha intentado ajustar los parámetros internos con los métodos que de optimización. Al principio se optó por utilizar GridSearchCV, pero los tiempos de computación eran excesivamente largos y no resultaba posible en el plazo existente. Por ello se decidió finalmente por el uso de RandomizedSearchCV, que es menos exacto, pero mucho más veloz.
A continuación, se expresan todas las pruebas realizadas con diferentes algoritmos. Los códigos a continuación han sido utilizados de forma discrecional según se iban evaluando las necesidades.

DEFINICIONES
GridSearchCV
Búsqueda de parámetros exhaustiva para un estimador. Se ajustan mediante una cuadrícula de valores, que va probando. Puede ser muy costoso computacionalmente. Por dentro usa además cross-validation.

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

RandomizedSearchCV
A diferencia del gridsearhcv, no se prueban todos los valores de los parámetros, si no que se muestra un número fijo de configuraciones. Mucho más rápido para pruebas.

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html

Stacking
Es una forma de agrupar varios modelos de regresión o clasificación. En nuestro caso serían de regresión. Es una buena forma de crear un modelo más preciso y consistente.

http://rasbt.github.io/mlxtend/user_guide/regressor/StackingCVRegressor/

http://rasbt.github.io/mlxtend/user_guide/regressor/StackingRegressor/

XGBOOST
Librería de gradient boosting con gran eficiencia. Esta implementado bajo el framework de Gradient Boosting. Este algoritmo es muy usado en problemas de regresión en kaggle.
https://xgboost.readthedocs.io/en/latest/
RandomForestRegressor
Se basa en la realización de árboles aleatorios. Se generan n árboles de decisión con varias submuestras de datos y se utiliza el promedio para ajustar la predicción, además de evitar el overffiting.
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
AdaBoostRegressor
Este tipo de regresor empieza ajustando el conjunto de datos original, para pasar más tarde a ajustar diferentes instancias de acuerdo con el error de predicción obtenido.
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
Epsilon-Support Vector Regression (SVR)¶
Intentan separar conjuntos de datos con hiperplanos para lograr la predicción.
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
Redes Neuronales (Sequential Model)¶
Conjunto de unidades conectadas entre sí para transmitir señales. La información de entrada atraviesa la red produciendo ciertos valores de salida.
https://es.wikipedia.org/wiki/Red_neuronal_artificial https://keras.io/models/sequential/
AI.2 Busqueda de Percentil Óptimo
Tras los distintos análisis exploratorios se detecta un considerable número de valores elevados que perjudican considerablemente al modelo. Con el objetivo de reducir el ruido y logar una mejor predicción sobre los valores normales, se opta por entrenar separando un percentil superior del dataset. Para seleccionar el percentil a separar del dataset se ha implementado un cross-validation manual que itera los distintos percentiles y almacena los resultados.
Tras ejecutar la iteración completa se obtiene el mejor resultado de MAE de forma consistente en el percentil 0.766.
AI.3 Natural Languaje Processing (NLP)
Como se ha comentado en el cuerpo del proyecto, con el objetivo de realizar un análisis detallado de los features de texto abierto, se ha empleado un procedimiento sistemático de clasificación y selección de palabras clave mediante "bag of words". Los resultados del "bag of words" se han correlacionado con la variable TARGET para obtener aquellas palabras clave más relevantes. Posteriormente, se han combinado estas palabras con las recomendaciones establecidas en la web enlazada de idealista comprendiendo el significado que tienen para los usuarios del portal.

https://www.idealista.com/news/inmobiliario/vivienda/2017/11/24/749048-consejos-practicos-para-conseguir-una-buena-descripcion-de-los-inmuebles
Todo esto ha resultado en el mejor modelo de análisis NLP.





