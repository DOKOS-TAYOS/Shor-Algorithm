"""

Author: Alejandro Mata Ali

Date: 16-05-2022

numpy (1.21.5): basic library for certain constants such as 𝜋 or functions for calculating binary numbers (floor and log2).
pandas (1.4.2): library for processing results in tabular form to improve analysis.
matplotlib.pyplot (3.5.1): package for graphing.
scipy (1.7.3): library for fitting simulated data to functions.
qat.lang.AQASM (2.1.3): module for the creation and execution of quantum circuits.
qat.qpus : package to obtain the qpu.
QQuantLib: the modules phase_estimation_wqft and iterative_quantum_pe from the PhaseEstimation package, and solver and data_extracting from the Utils package of this library. They can be found at the GitHub link https://github.com/NEASQC/FinancialApplications. In addition we use the Shor module for the functions implemented in the previous notebook.

Reference: https://arxiv.org/abs/quant-ph/0205095

"""

#--------------------------------------------------

#Visualizacion y calculo.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Cuantico
from qat.lang.AQASM import *
from qat.qpus import get_default_qpu
#Libreria externa para realizar la QFT
from QQuantLib.PE.phase_estimation_wqft import PhaseEstimationwQFTaprox
from QQuantLib.PE.iterative_quantum_pe import IterativeQuantumPE
from QQuantLib.utils.data_extracting import get_results
from QQuantLib.utils.qlm_solver import get_qpu

#--------------------------------------------------
#Functions

def numbin(a: int, n: int) -> str:
    '''
    Function that converts the decimal number a into a binary number with n bits.
    '''
    a = format( a, 'b' )#Pass the number to binary
    return '0'*(n-len(a)) + a#We add the necessary leading zeros to make it have the number of bits.

def m_inv(a: int, N: int) -> int:
    '''
    Function that implements the inverse module of N.
    '''
    for _ in range(1, N):#We run through all the numbers between 1 and N.
        if ((a%N) * (_%N)) % N == 1: return _#If it is the inverse, we return the result and we are done.
    raise Exception('La inversa modulo no existe.')# In case there is no such number.

def primador(p: int, q: int, itmax: int = 100, used_a: list = []):
    '''Function that implements the creation of a number N from 2 primes
    and gives us a valid number a for the algorithm.
    
    Parameters:
    ----------
    
    itmax: the number of iterations allowed to search for this a.
    
    used_a: values of a that we do not want to have as a result.
    
    '''
    N   = p * q
    ver = True; itera = 0      #Check that the greatest common divisor of a and N is 1.
    while ver and itera<itmax:
        a  = np.random.randint(2, N)#Random number between 2 and N.
        cd = np.gcd(a, N)
        if cd == 1 and (a not in used_a): ver = False
        itera += 1#Iteration counter.
    if ver == True: print('No valid a has been found.')
    return a, N

def reductor(num: int, den: int):
    '''
    Function that implements the obtaining of 
    the irreducible fraction of a number.
    The input number will be a fraction num/den.
    It is based on dividing by all possible values.
    '''
    for i in range(num, 0, -1):
        if num % i == 0 and den % i == 0:
            num /= i; den /= i
    return int(num), int(den)

#--------------------------------------------------
#Gates

@build_gate('QFT_ap', [int, int], arity=lambda n, aprox: n)
def QFT_aprox(n, aprox):
    '''Function implementing the approximate Quantum Fourier Transformation.
    Its input parameters are the number of qbits involved n and the approximate maximum distance we allow for the controlled phase gates.
    we allow for the controlled phase gates.'''
    routine = QRoutine()
    
    x = routine.new_wires(n)
    for i in range(n):
        routine.apply(H, x[i])
        for j in range(i+1, n):
            if (j-i) < aprox:
                routine.apply(PH(np.pi/2**(j-i)).ctrl(), x[j], x[i])
            
    return routine

@build_gate('ADD', [int, str], arity=lambda n, a: n)
def add_g(n, a):
    '''Implementation of the phiADD(a) gate with n qbits.
    phi(b) -> phi(b+a)
    a is the binary of the number to implement.'''
    #-------------------------------------
    routine = QRoutine()
    #-------------------------------------
    b = routine.new_wires(n)#Input lines.
    
    #-------------------------------------
    #We executed the addition.
    for i in range(n):#Target
        angle = sum( np.array( [ int(a[j])/2**(j-i) for j in range(i, n) ] ) )
        if angle != 0: routine.apply( PH(np.pi*angle), b[i] )
    
    return routine


@build_gate('ADDmod_ap', [int, str, str, int], arity=lambda n, a, N, aprox: n)
def addmod_g_ap(n, a, N, aprox):
    '''Implementation of the phiADD(a)mod(N) gate with n qbits.
    phi(b) -> phi((b+a)mod(N))
    a and N are the binary numbers to be implemented.'''
    #-------------------------------------
    routine = QRoutine()
    #-------------------------------------
    b  = routine.new_wires(n)#Input lines
    a0 = routine.new_wires(1); routine.set_ancillae(a0)#Control ancilla of the most significant bit.
    
    #-------------------------------------
    #We apply the algorithm:
    #Sum a, subtract N, QFT-1, we take out the most significant, QFT, sum N.
    routine.apply(add_g(n, a), b)
    
    routine.apply(add_g(n, N).dag(), b)
    routine.apply(QFT_aprox(n, aprox).dag(), b)
    routine.apply(CNOT, b[0], a0)
    routine.apply(QFT_aprox(n, aprox), b)
    routine.apply(add_g(n, N).ctrl(), a0, b)
    
    
    #-------------------------------------
    #Subtract a, QFT-1, significant, QFT, sum a
    routine.apply(add_g(n, a).dag(), b)
    
    routine.apply(QFT_aprox(n, aprox).dag(), b)
    routine.apply(X, b[0]); routine.apply(CNOT, b[0], a0); routine.apply(X, b[0])
    routine.apply(QFT_aprox(n, aprox), b)
    
    routine.apply(add_g(n, a), b)
    
    return routine

@build_gate('CMULT_ap', [int, str, str, int], arity=lambda n, a, N, aprox: 2*n )
def cmult_g_ap(n, a, N, aprox):
    '''Implementation of the phiCMULT(a)mod(N) gate with n qbits.
    b -> (b+a*x)mod(N)
    a and N are the binary of the number to implement.
    Enter X first and then b'''
    #-------------------------------------
    routine = QRoutine()
    #-------------------------------------
    x = routine.new_wires(n)#Input x (first).
    b = routine.new_wires(n)#Input b (second).
    
    #-------------------------------------
    #Transform, set of sums, Antitransform.
    routine.apply(QFT_aprox(n, aprox), b)
    #We start at 0 (most significant) and work our way down.
    for i in range(0, n):#Control.
        for j in range(0, int(2**i)):#Number of applications.
            routine.apply( addmod_g_ap(n, a, N, aprox).ctrl(), x[-i-1], b )
    routine.apply(QFT_aprox(n, aprox).dag(), b)
    
    return routine

@build_gate('U_ap', [int, int, int, int], arity=lambda n, a, N, aprox: n)
def ua_g_ap(n, a, N, aprox):
    '''Implementation of the phiCMULT(a)mod(N) gate with n qbits.
    x -> (a*x)mod(N)
    a and n are integers.'''
    #-------------------------------------
    #Prepare the parameters and binaries.
    abin = numbin(a, n)
    ainv = numbin(m_inv(a, N), n)#a inverse modulo N
    Nbin = numbin(N, n)
    
    #-------------------------------------
    routine = QRoutine()
    #-------------------------------------
    x  = routine.new_wires(n)#Input lines
    a0 = routine.new_wires(n); routine.set_ancillae(a0)#Ancillas at 0.
    
    #-------------------------------------
    #We apply the algorithm:
    #Multiply, swap, divide.
    routine.apply(cmult_g_ap(n, abin, Nbin, aprox), x+a0)
    for i in range(n): routine.apply(SWAP, x[i], a0[i])
    routine.apply(cmult_g_ap(n, ainv, Nbin, aprox).dag(), x+a0)
    
    return routine

#--------------------------------------------------
#Ejecutador final

def Shor_ap(p: int, q: int, shots: int = 8, method: str = 'QPE', used_a: list = [], aprox: list = [3, 3], a: int = 0, QLMaaS = True):
    '''
    Function that implements all of Shor's algorithm for a given pair p, q of
    given integer prime numbers.
    
    Parameters:
    ----------
    
    p, q: prime numbers that we are going to multiply to get N.
    
    shots: number of measurements of the circuit.
    
    method: resolution method to use.
    If we put QPE it will use the quantum phase estimation with QFT.
    If we put IQPE it will use the Iterative Quantum Phase Estimation.
    
    used_a: values of a that we do not want to be used.
    
    a: predefined value of a. If left at 0, it will search for an a that meets the algorithm conditions.
    meets the conditions of the algorithm.
    
    QLMaaS: QLMaaS == False -> PyLinalg
            QLMaaS == True -> Try to use LinAlg (To use QPU as the CESGA QLM)
            
    aprox: approximation we are going to use for ADD and QFT, respectively.
    '''
    
    #-------------------------------------
    if a == 0: a, N = primador(p, q, used_a = used_a)
    else: N = p * q
    #Determinamos el numero de qbits.
    n = 2 + int(np.floor(np.log2(N)))
    n_cbits = n #El mismo para los clasicos.
    print('a = ', a, ', N = ', N, ', n = ', n, ', aprox = ', aprox)
    print('Number of qbits Gate: ', 2*n+1)
    print('Total number of qbits: ', 4*n+1)
    
    #Our Shor's Gate.
    unitary_operator = ua_g_aprox(n, a, N, aprox[0], aprox[1])
    #-------------------------------------
    #Initial state.
    initial_state = QRoutine()
    x = initial_state.new_wires(n)
    initial_state.apply(X, x[-1])#State 1.
    
    
    #Python dictionary for configuration.
    config_dict = {
    'initial_state': initial_state,
    'unitary_operator': unitary_operator,
    'qpu' : linalg_qpu,
    'auxiliar_qbits_number' : n_cbits,  
    'shots': shots,
    'aprox': aprox[1]
    }
    
    if method == 'QPE':
        qft_pe = PhaseEstimationwQFTaprox(**config_dict)
        start = time.time()
        qft_pe.pe_wqft()#We send it to execute
        stop = time.time()
        print('Execution time: ', stop - start)
        Resultados = qft_pe.final_results.loc[qft_pe.final_results['Probability'] > 0]
        Valores    = [ int(_) for _ in Resultados['Int'] ]
        Probabilidad = np.array([ _ for _ in Resultados['Probability'] ])*100#Because it is a percentage
    
    elif method == 'IQPE':
        iqpe_ = IterativeQuantumPE(**config_dict)
        start = time.time()
        iqpe_.iqpe()#We send it to execute
        stop = time.time()
        print('Execution time: ', stop - start)
        Resultados = iqpe_.sumarize(iqpe_.final_results, 'BitInt')
        Valores    = [ int(_) for _ in Resultados['BitInt'] ]
        Probabilidad = np.array([ _ for _ in Resultados['Frequency'] ])
        Probabilidad = Probabilidad * 100 / sum(Probabilidad)#Because it is a percentage
    
    check = 0
    rs = []#List of r obtained.
    r_correct = []#List of correct r's
    for i in range(len(Valores)):#0 does not contribute.
        if Valores[i] != 0:
            r = reductor(Valores[i], 2**n)[1]#The denominator.
            if r not in rs and r % 2 == 0:
                if a**r != -1 % N:
                    rs.append(r)
                    p_obt = np.gcd(int(a**int(r/2) - 1), N);   q_obt = np.gcd(int(a**int(r/2) + 1), N)
                    print('r = ', r, end=', ')
                    if (p == q_obt or q == p_obt) or (p == p_obt or q == q_obt):
                        print('The primes are: ', p_obt, ', ', q_obt)
                        r_correct.append(r); check += Probabilidad[i]
                    else: print('Not satisfied. The primes obtained are: ', p_obt, q_obt)
            elif r in r_correct: check += Probabilidad[i] #For repetitions.
    if check == 0: p_obt = 0; q_obt = 0; print('Not found.')
    print('Hits = ', check, ' %')
    return Valores, p_obt, q_obt, check, a

