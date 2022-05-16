"""

Autor: Alejandro Mata Ali

Date: 16-05-2022

numpy (1.21.5): basic library for certain constants such as pi or functions for calculating binary numbers (floor and log2).
pandas (1.4.2): library for processing results in tabular form to improve analysis.
qat.lang.AQASM (2.1.3): module for the creation and execution of quantum circuits.
qat.qpus : package to obtain the qpu.
QQuantLib: the modules phase_estimation_wqft and iterative_quantum_pe from the PhaseEstimation package, and solver and data_extracting from the Utils package of this library. It can be found at the GitHub link https://github.com/NEASQC/FinancialApplications.

Reference: https://arxiv.org/abs/quant-ph/0205095
"""

#--------------------------------------------------

import numpy as np
import pandas as pd
import time
#Quantum
from qat.lang.AQASM import *
from qat.qpus import get_default_qpu
#External library to perform QFT
from QQuantLib.PE.phase_estimation_wqft import PhaseEstimationwQFT
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

@build_gate('ADD', [int, str], arity=lambda n, a: n)
def add_g(n: str, a: int):
    '''
    Implementation of the phiADD(a) gate with n qbits.
    phi(b) -> phi(b+a)
    
    Parameters:
    ----------
    
    a: binary of the number to implement.
    
    n: number of qbits to use.
    '''
    #-------------------------------------
    routine = QRoutine()
    #-------------------------------------
    b = routine.new_wires(n)#Input lines
    
    #-------------------------------------
    #We executed the addition.
    for i in range(n):#Target
        angle = sum( np.array( [ int(a[j])/2**(j-i) for j in range(i, n) ] ) )
        if angle != 0: routine.apply( PH(np.pi*angle), b[i] )
    
    return routine


@build_gate('ADDmod', [int, str, str], arity=lambda n, a, N: n)
def addmod_g(n: int, a: str, N: int):
    '''
    Implementation of the phiADD(a)mod(N) gate with n qbits.
    phi(b) -> phi((b+a)mod(N))
    a and N are the binary numbers to be implemented.
    '''
    #-------------------------------------
    routine = QRoutine()
    #-------------------------------------
    b  = routine.new_wires(n)#Input lines.
    a0 = routine.new_wires(1); routine.set_ancillae(a0)#Control ancilla of the most significant bit.
    
    #-------------------------------------
    #We apply the algorithm:
    #Sum a, subtract N, QFT-1, we take out the most significant, QFT, sum N.
    routine.apply(add_g(n, a), b)
    
    routine.apply(add_g(n, N).dag(), b)
    routine.apply(qftarith.QFT(n).dag(), b)
    routine.apply(CNOT, b[0], a0)
    routine.apply(qftarith.QFT(n), b)
    routine.apply(add_g(n, N).ctrl(), a0, b)
    
    
    #-------------------------------------
    #Subtract a, QFT-1, significant, QFT, sum a
    routine.apply(add_g(n, a).dag(), b)
    
    routine.apply(qftarith.QFT(n).dag(), b)
    routine.apply(X, b[0]); routine.apply(CNOT, b[0], a0); routine.apply(X, b[0])
    routine.apply(qftarith.QFT(n), b)
    
    routine.apply(add_g(n, a), b)
    
    return routine


@build_gate('CMULT', [int, str, str], arity=lambda n, a, N: 2*n )
def cmult_g(n: int, a: str, N: str):
    '''
    Implementation of the phiCMULT(a)mod(N) gate with n qbits.
    b -> (b+a*x)mod(N)
    a and N are the binary numbers to be implemented.
    When calling it, we will have to do x+b in the gate target
    to respect the internal order of operations.
    '''
    #-------------------------------------
    routine = QRoutine()
    #-------------------------------------
    x = routine.new_wires(n)#Input x (first).
    b = routine.new_wires(n)#Entry b (second).
    
    #-------------------------------------
    #Transform, set of sums, Antitransform.
    routine.apply(qftarith.QFT(n), b)
    #We start at 0 (plus signif) and work our way down.
    for i in range(0, n):#Control.
        for j in range(0, int(2**i)):#Number of applications.
            routine.apply( addmod_g(n, a, N).ctrl(), x[-i-1], b )
    routine.apply(qftarith.QFT(n).dag(), b)
    
    return routine


@build_gate('U', [int, int, int], arity=lambda n, a, N: n)
def ua_g(n: int, a: int, N: int):
    '''
    Implementation of the phiCMULT(a)mod(N) gate with n qbits.
    x -> (a*x)mod(N).
    '''
    #-------------------------------------
    #Prepare the parameters and binaries.
    abin = numbin(a, n)
    ainv = numbin(m_inv(a, N), n)#a inverse modulo N
    Nbin = numbin(N, n)
    
    #-------------------------------------
    routine = QRoutine()
    #-------------------------------------
    x  = routine.new_wires(n)#Input lines.
    a0 = routine.new_wires(n); routine.set_ancillae(a0)#Ancillas at 0.
    
    #-------------------------------------
    #We apply the algorithm:
    #Multiply, swap, divide.
    routine.apply(cmult_g(n, abin, Nbin), x+a0)
    for i in range(n): routine.apply(SWAP, x[i], a0[i])
    routine.apply(cmult_g(n, ainv, Nbin).dag(), x+a0)
    
    return routine

#--------------------------------------------------
#Final performer

def Shor(p: int, q: int, shots: int = 8, method: str = 'QPE', used_a: list = [], a: int = 0, QLMaaS = True):
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
    '''
    
    #-------------------------------------
    if a == 0: a, N = primador(p, q, used_a = used_a)
    else: N = p * q
    #We determine the number of qbits.
    n = 2 + int(np.floor(np.log2(N)))
    n_cbits = n #Same for the classics.
    print('a = ', a, ', N = ', N, ', n = ', n)
    print('Numero de qbits Puerta: ', 2*n+1)
    print('Numero de qbits total: ', 4*n+1)
    
    #Our Gate of Shor.
    unitary_operator = ua_g(n, a, N)
    #-------------------------------------
    #Initial state.
    initial_state = QRoutine()
    x = initial_state.new_wires(n)
    initial_state.apply(X, x[-1])#State 1.
    
    linalg_qpu = get_qpu(QLMaaS)
    #Python dictionary for configuration.
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
        qft_pe.pe_wqft()#We send it to execute
        stop = time.time()
        print('Tiempo de ejecución: ', stop - start)
        Resultados = qft_pe.final_results.loc[qft_pe.final_results['Probability'] > 0]
        Valores    = [ int(_) for _ in Resultados['Int'] ]
        Probabilidad = np.array([ _ for _ in Resultados['Probability'] ])*100#Because it is a percentage
    
    elif method == 'IQPE':
        iqpe_ = IterativeQuantumPE(**config_dict)
        start = time.time()
        iqpe_.iqpe()#We send it to execute
        stop = time.time()
        print('Tiempo de ejecución: ', stop - start)
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
                if int(a**(r/2)) != -1 % N:
                    rs.append(r)
                    p_obt = np.gcd(int(a**int(r/2) - 1), N);   q_obt = np.gcd(int(a**int(r/2) + 1), N)
                    print('r = ', r, end=', ')
                    if (p == q_obt or q == p_obt) or (p == p_obt or q == q_obt):
                        print('The primes are: ', p_obt, ', ', q_obt)
                        r_correct.append(r); check += Probabilidad[i]
                    else: print('Not satisfied. The primes obtained are: ', p_obt, q_obt)
            elif r in r_correct: check += Probabilidad[i] #Because of the repetitions.
    if check == 0: p_obt = 0; q_obt = 0; print('Not found.')
    print('Hits = ', check, ' %')
    return Valores, p_obt, q_obt, check, a

