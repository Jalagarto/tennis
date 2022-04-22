
EDA: Exploratory data analysis is an interesting preprocessing step to applying ML algorithms.  
It is recommended to do it in almost any ML project, so we can understand the nature
of the features (variables) and the relationships between them.

It is also recommended so ML algorithms can learn from data faster and better.

Discrete vars:
[setNumber, gameNumber, pointNumber, serveNumber, shotCount, efectividad,
1º o 2º saque, seve_class, Lado(1:iguales, 0:ventaja), DIRECCIÓN, X1,...]

Discrete vars to delete (only one value) --> no information!
[Point,

Continuous vars:
v(m/s), v(km/h), timeHIT,Y1, Y1(ABS), Z1(h), timeNET, TimeN-TimeH,
timeBounce, TimeB-TimeH,PreVx, PreVy, PreVz, PreV, PostVx, PostVy,
PostVz, PostV, Dif. V1-V2, Dif(km/h)


Vars to delete (duplicated):
- v(km/h), Dif. (km/h), Point,  


All vars:
['1º o 2º saque',
 'DIRECCIÓN:1 abierto;2 al cuerpo;3 a la T',
 'Dif. (km/h)',
 'Dif. V1-V2',
 'Efectividad',
 'EventID',
 'Lado(1:Iguales;0:Ventaja)',
 'Point',
 'PostV',
 'PostVx',
 'PostVy',
 'PostVz',
 'PreV',
 'PreVx',
 'PreVy',
 'PreVz',
 'Sexo',
 'TimeB-TimeH',
 'TimeN-TimeH',
 'V (km/h)',
 'V (m/s)',
 'X1',
 'Y1',
 'Y1(ABS)',
 'Z1 (h)',
 'gameNumber',
 'pointNumber',
 'serveNumber',
 'serve_class',
 'setNumber',
 'shotCount',
 'startScore_Receiver',
 'startScore_Server',
 'timeBounce',
 'timeHIT',
 'timeNET']
