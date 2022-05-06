Works performed:
----------------

Os comento por encima los trabajos realizados:  

1. EDA y preprocesamiento de datos (ya visto en primer email)  

2. Entrenamiento de múltiples modelo de clasificación (binaria). Se entrenaron unos 20 modelos y se consiguió una "accuracy"
o exactitud de la precisión del 89%. Es decir, tenemos un modelo capaz de predecir si el saque va a ser efectivo o no (con un
89% de exactitud).  

3. Cálculo de un modelo probabilístico a partir del modelo inicial. Esto se hizo porqué un output de 1s y 0s es algo menos informativo que si sabemos la probabilidad de que sea efectivo (nos deja ver los differentes tonos de grises, en vez de un B&W).  

4. Cálculo de Feature importance --> nos dice qué importancia tiene cada variable en la efectividad  

5. Herramienta para probar diferentes valores de las variables. Esto es un juguete para que probéis a analizar variables.
Se ha generado utilizando las variables más relevantes, pero puede hacerse con más variables  

Hay alguna cosa que no me cuadra, sin embargo. Entiendo que todos los ACES y los NOT RETURNED tienen efectividad = 1 y los saques FAULT tendrían efectividad = 0? quizá deberíamos hacer un modelo que únicamente utilice los RETURNED?  


Es decir, los outputs de este ejercicio son:  

1. Modelo de Classificación de efectividad  

2. Cálculo de la importancia de las variables a partir del modelo entrenado  

3. Herramienta que permite analizar las diferentes variables  

Resultados de evaluación del modelo:

