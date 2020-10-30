# Ecuaciones Lineales

## cholesky.py

Hace la descomposición Cholesky y devuelve las dos matrices.

Depende de: 

_Librerías_

- numpy

- [x] Tests.

> NOTA: Los tests son "a mano". Recordar que a Fierens mucho no le copa esto.

-----------------------------------------------------------------------------

## DVS.py

Realiza descomposición en valores singulares con la función svd() de una matrix
R, con dimensiones mxn.


Depende de: 

_Funciones:_

- _sortMatrices()
- _norm()
- _projectOperator()
- _matrixOrthonormalization()

_Librerías_

- numpy
- random

Tiene una función llamada `psuedoInverse` que calcula la matriz psuedo-inversa
(también llamada inversa de Moore-Penrose).

Para cuadrados mínimos, (la matriz pseudo inversa está notada como: A+)

x = A+ * b


**NO USAR** para cuadrados mínimos a menos que la solución del sistema no sea
única (es muy costoso). En dicho caso, este es el único algoritmo posible.

Es decir, esta es la **opción ideal cuando**  el rango de la matriz que
queremos usar es menor a su número de columnas.


Depende de:

_Funciones:_

- svd()


- [ ] ~~Tests para `svd`~~
- [ ] ~~Tests para `pseudoInverse`~~

-----------------------------------------------------------------------------

## iteration.py



-----------------------------------------------------------------------------

## LUdec.py



-----------------------------------------------------------------------------

## qr.py



-----------------------------------------------------------------------------

## subsBackwards.py

Realiza sustitución hacia atrás. Útil para resolver sistemas con una matriz
**triangular superior**.


> Ejemplo de triangular superior:
>
> |  2  5  6 | <br>
> |  0  1  4 | <br>
> |  0  0  9 |

Depende de: 

_Librerías_

- numpy

- [x] Tests.

> NOTA: Los tests son "a mano". Recordar que a Fierens mucho no le copa esto.

-----------------------------------------------------------------------------

## subsForwards.py

Realiza sustitución hacia adelante. Útil para resolver sistemas con una matriz
**triangular inferior**.


> Ejemplo de triangular inferior:
>
> |  2   0  0 | <br>
> |  4   1  0 | <br>
> |  6   4  9 |


Depende de: 

_Librerías_

- numpy

- [x] Tests.

> NOTA: Los tests son "a mano". Recordar que a Fierens mucho no le copa esto.

-----------------------------------------------------------------------------
