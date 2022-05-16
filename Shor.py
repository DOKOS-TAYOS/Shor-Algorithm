"""

Autor: Alejandro Mata Ali

Fecha: 16-05-2022

numpy (1.21.5): librería básica para ciertas constantes como  pi  o funciones para el cálculo de números binarios (floor y log2).
pandas (1.4.2): librería para el procesamiento de resultados en forma de tabla y mejorar los análisis.
qat.lang.AQASM (2.1.3): módulo para la creación y ejecución de los circuitos cuánticos.
qat.qpus : paquete para obtener la qpu.
QQuantLib: los módulos phase_estimation_wqft y iterative_quantum_pe del paquete PhaseEstimation, y solver y data_extracting del paquete Utils de esta librería. Se puede encontrar en el enlace de GitHub https://github.com/NEASQC/FinancialApplications.

Referencia: https://arxiv.org/abs/quant-ph/0205095

"""

#--------------------------------------------------

import numpy as np
import pandas as pd
import time
#Cuantico
from qat.lang.AQASM import *
from qat.qpus import get_default_qpu
#Libreria externa para realizar la QFT
from QQuantLib.PE.phase_estimation_wqft import PhaseEstimationwQFT
from QQuantLib.PE.iterative_quantum_pe import IterativeQuantumPE
from QQuantLib.utils.data_extracting import get_results
from QQuantLib.utils.qlm_solver import get_qpu

#--------------------------------------------------
#Funciones

def numbin(a: int, n: int) -> str:
    '''
    Funcion que convierte el numero decimal a en uno binario con n bits.
    '''
    a = format( a, 'b' )#Pasamos el numero a binario
    return '0'*(n-len(a)) + a#Añadimos los ceros a la izquierda necesarios para que tenga el numero de bits.

def m_inv(a: int, N: int) -> int:
    '''
    Funcion que implementa a modulo inverso de N.
    '''
    for _ in range(1, N):#Recorremos todos los numeros entre 1 y N.
        if ((a%N) * (_%N)) % N == 1: return _#Si es el inverso, devolvemos el resultado y acabamos.
    raise Exception('La inversa modulo no existe.')# Por si no existiese dicho numero.

def primador(p: int, q: int, itmax: int = 100, used_a: list = []):
    '''Funcion que implementa el crear un numero N a partir de 2 primos
    y nos da un numero a valido para el algoritmo.
    
    Parametros:
    ----------
    
    itmax:  el numero de iteraciones permitidas para buscar este a.
    
    used_a: valores de a que no queremos tener como resultado.
    
    '''
    N   = p * q
    ver = True; itera = 0      #Comprueba que el maximo comun divisor de a y N sea 1.
    while ver and itera<itmax:
        a  = np.random.randint(2, N)#Entero aleatorio entre 2 y N.
        cd = np.gcd(a, N)
        if cd == 1 and (a not in used_a): ver = False
        itera += 1#Contador de iteraciones.
    if ver == True: print('No se ha encontrado un a valido.')
    return a, N

def reductor(num: int, den: int):
    '''
    Funcion que implementa la obtencion de 
    la fraccion irreducible de un numero.
    El número de entrada va a ser una fraccion num/den.
    Se basa en dividir entre todos los valores posibles.
    '''
    for i in range(num, 0, -1):
        if num % i == 0 and den % i == 0:
            num /= i; den /= i
    return int(num), int(den)

#--------------------------------------------------
#Puertas

@build_gate('ADD', [int, str], arity=lambda n, a: n)
def add_g(n: str, a: int):
    '''
    Implementacion de la puerta phiADD(a) con n qbits.
    phi(b) -> phi(b+a)
    
    Parametros:
    ----------
    
    a: binario del numero a implementar.
    
    n: numero de qbits a usar.
    '''
    #-------------------------------------
    routine = QRoutine()
    #-------------------------------------
    b = routine.new_wires(n)#Lineas de entrada.
    
    #-------------------------------------
    #Ejecutamos la adiccion.
    for i in range(n):#Objetivo
        angle = sum( np.array( [ int(a[j])/2**(j-i) for j in range(i, n) ] ) )
        if angle != 0: routine.apply( PH(np.pi*angle), b[i] )
    
    return routine


@build_gate('ADDmod', [int, str, str], arity=lambda n, a, N: n)
def addmod_g(n: int, a: str, N: int):
    '''
    Implementacion de la puerta phiADD(a)mod(N) con n qbits.
    phi(b) -> phi((b+a)mod(N))
    a y N son los binarios del numero a implementar.
    '''
    #-------------------------------------
    routine = QRoutine()
    #-------------------------------------
    b  = routine.new_wires(n)#Lineas de entrada.
    a0 = routine.new_wires(1); routine.set_ancillae(a0)#Ancilla de control del bit mas significativo.
    
    #-------------------------------------
    #Aplicamos el algoritmo:
    #Suma a, resta N, QFT-1, sacamos el mas significativo, QFT, suma N
    routine.apply(add_g(n, a), b)
    
    routine.apply(add_g(n, N).dag(), b)
    routine.apply(qftarith.QFT(n).dag(), b)
    routine.apply(CNOT, b[0], a0)
    routine.apply(qftarith.QFT(n), b)
    routine.apply(add_g(n, N).ctrl(), a0, b)
    
    
    #-------------------------------------
    #Resta a, QFT-1, significativo, QFT, suma a
    routine.apply(add_g(n, a).dag(), b)
    
    routine.apply(qftarith.QFT(n).dag(), b)
    routine.apply(X, b[0]); routine.apply(CNOT, b[0], a0); routine.apply(X, b[0])
    routine.apply(qftarith.QFT(n), b)
    
    routine.apply(add_g(n, a), b)
    
    return routine


@build_gate('CMULT', [int, str, str], arity=lambda n, a, N: 2*n )
def cmult_g(n: int, a: str, N: str):
    '''
    Implementacion de la puerta phiCMULT(a)mod(N) con n qbits.
    b -> (b+a*x)mod(N)
    a y N son los binarios del numero a implementar.
    Al llamarla, deberemos hacer x+b en el objetivo de la puerta
    para respetar el orden interno de las operaciones.
    '''
    #-------------------------------------
    routine = QRoutine()
    #-------------------------------------
    x = routine.new_wires(n)#Entrada x (primero).
    b = routine.new_wires(n)#Entrada b (segundo).
    
    #-------------------------------------
    #Transformada, conjunto de sumas, Antitransformada.
    routine.apply(qftarith.QFT(n), b)
    #Empezamos en el 0 (mas signif) y vamos bajando.
    for i in range(0, n):#Control.
        for j in range(0, int(2**i)):#Numero de aplicaciones.
            routine.apply( addmod_g(n, a, N).ctrl(), x[-i-1], b )
    routine.apply(qftarith.QFT(n).dag(), b)
    
    return routine


@build_gate('U', [int, int, int], arity=lambda n, a, N: n)
def ua_g(n: int, a: int, N: int):
    '''
    Implementacion de la puerta phiCMULT(a)mod(N) con n qbits.
    x -> (a*x)mod(N).
    '''
    #-------------------------------------
    #Preparamos los parametros y los binarios.
    abin = numbin(a, n)
    ainv = numbin(m_inv(a, N), n)#a inverso modulo N
    Nbin = numbin(N, n)
    
    #-------------------------------------
    routine = QRoutine()
    #-------------------------------------
    x  = routine.new_wires(n)#Linea de entrada.
    a0 = routine.new_wires(n); routine.set_ancillae(a0)#Ancillas a 0.
    
    #-------------------------------------
    #Aplicamos el algoritmo:
    #Multiplicamos, swap, dividimos.
    routine.apply(cmult_g(n, abin, Nbin), x+a0)
    for i in range(n): routine.apply(SWAP, x[i], a0[i])
    routine.apply(cmult_g(n, ainv, Nbin).dag(), x+a0)
    
    return routine

#--------------------------------------------------
#Ejecutador final

def Shor(p: int, q: int, shots: int = 8, method: str = 'QPE', used_a: list = [], a: int = 0, QLMaaS = True):
    '''
    Funcion que implementa todo el algoritmo de Shor para un par p, q de
    numeros primos enteros dados.
    
    Parametros:
    ----------
    
    p, q: numeros primos que vamos a multiplicar para obtener N.
    
    shots: numero de mediciones del circuito.
    
    method: metodo de resolucion a usar.
    Si ponemos QPE usara el quantum phase estimation with QFT.
    Si ponemos IQPE usara el Iterative Quantum Phase Estimation.
    
    used_a: valores de a que no queremos que se usen.
    
    a: valor predefinido de a. Si se deja a 0, se buscara un a que
    cumpla las condiciones del algoritmo.
    
    QLMaaS: QLMaaS == False -> PyLinalg
            QLMaaS == True -> Intenta usar LinAlg (Para usar QPU como el CESGA QLM)
    '''
    
    #-------------------------------------
    if a == 0: a, N = primador(p, q, used_a = used_a)
    else: N = p * q
    #Determinamos el numero de qbits.
    n = 2 + int(np.floor(np.log2(N)))
    n_cbits = n #El mismo para los clasicos.
    print('a = ', a, ', N = ', N, ', n = ', n)
    print('Numero de qbits Puerta: ', 2*n+1)
    print('Numero de qbits total: ', 4*n+1)
    
    #Nuestra puerta de Shor.
    unitary_operator = ua_g(n, a, N)
    #-------------------------------------
    #Estado inicial.
    initial_state = QRoutine()
    x = initial_state.new_wires(n)
    initial_state.apply(X, x[-1])#Estado 1.
    
    linalg_qpu = get_qpu(QLMaaS)
    #Diccionario de Python para la configuracion.
    config_dict = {
    'initial_state': initial_state,
    'unitary_operator': unitary_operator,
    'qpu' : linalg_qpu,
    'auxiliar_qbits_number' : n_cbits,  
    'shots': shots
    }
    
    if method == 'QPE':
        qft_pe = PhaseEstimationwQFT(**config_dict)
        start = time.time()
        qft_pe.pe_wqft()#La enviamos a ejecutar
        stop = time.time()
        print('Tiempo de ejecución: ', stop - start)
        Resultados = qft_pe.final_results.loc[qft_pe.final_results['Probability'] > 0]
        Valores    = [ int(_) for _ in Resultados['Int'] ]
        Probabilidad = np.array([ _ for _ in Resultados['Probability'] ])*100#Por ser porcentual
    
    elif method == 'IQPE':
        iqpe_ = IterativeQuantumPE(**config_dict)
        start = time.time()
        iqpe_.iqpe()
        stop = time.time()
        print('Tiempo de ejecución: ', stop - start)
        Resultados = iqpe_.sumarize(iqpe_.final_results, 'BitInt')
        Valores    = [ int(_) for _ in Resultados['BitInt'] ]
        Probabilidad = np.array([ _ for _ in Resultados['Frequency'] ])
        Probabilidad = Probabilidad * 100 / sum(Probabilidad)
    
    check = 0
    rs = []#Lista de r obtenidos.
    r_correct = []#Lista de r correctos
    for i in range(len(Valores)):#El 0 no contribuye.
        if Valores[i] != 0:
            r = reductor(Valores[i], 2**n)[1]#El denominador.
            if r not in rs and r % 2 == 0:
                if int(a**(r/2)) != -1 % N:
                    rs.append(r)
                    p_obt = np.gcd(int(a**int(r/2) - 1), N);   q_obt = np.gcd(int(a**int(r/2) + 1), N)
                    print('r = ', r, end=', ')
                    if (p == q_obt or q == p_obt) or (p == p_obt or q == q_obt):
                        print('Los primos son: ', p_obt, ', ', q_obt)
                        r_correct.append(r); check += Probabilidad[i]
                    else: print('No se cumple. Los primos obtenidos son: ', p_obt, q_obt)
            elif r in r_correct: check += Probabilidad[i] #Por las repeticiones.
    if check == 0: p_obt = 0; q_obt = 0; print('No encontrados.')
    print('Aciertos = ', check, ' %')
    return Valores, p_obt, q_obt, check, a

