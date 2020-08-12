#!C:/PATH_TO_YOUR_QUANTUM_ENVIRONMENT/python.exe
print('Content-Type: application/json\n\n')



###############################
###### PROBLEM DEFINITION #####
###############################

## Cost function specification in sympy

import sympy as sym

# number of edges (Qubits)
N = sym.symbols('N')

# possible edges X_0, ..., X_{N-1} with values in {0,1}
X = sym.IndexedBase('X')

# costs for edges
C = sym.IndexedBase('C')

# penalty
P = sym.symbols('P')

# indices
i, j, k, v = sym.symbols('i j k v')

# start vertex: incoming and outgoing edges
VsI = sym.IndexedBase('VsI')    # incoming edges
LenVsI = sym.symbols('LenVsI')  # number of incoming edges
VsO = sym.IndexedBase('VsO')    # outgoing edges
LenVsO = sym.symbols('LenVsO')  # number of incoming edges

# target vertex: incoming and outgoing edges
VtI = sym.IndexedBase('VtI')    # incoming edges
LenVtI = sym.symbols('LenVtI')  # number of incoming edges
VtO = sym.IndexedBase('Vt O')   # outgoing edges
LenVtO = sym.symbols('LenVtO')  # number of incoming edges

# middle vertices: incoming and outgoing edges
# two indices: VI_(i,k) is the kth incoming edge at vertex i
# Similarly, LenVI_i is the number of incoming edges at vertex i
# the number of lists in LenVI and LenVO is the number of central vertices and must equal
LenV = sym.IndexedBase('LenV')   # number central vertices

VI = sym.IndexedBase('VI')        # incoming edges
LenVI = sym.IndexedBase('LenVI')  # number of incoming edges
VO = sym.IndexedBase('VO')        # outgoing edges
LenVO = sym.IndexedBase('LenVO')  # number of outgoing edges

cost_function = (
    sym.Sum( C[i] * X[i] , (i,0,N-1) ) +                    # costs for edge i
    P * (sym.Sum( X[VsO[i]], (i, 0, LenVsO-1) ) - 1 )**2 +  # exactly one outgoing edge at start vertex
    P * sym.Sum( X[VsI[i]], (i, 0, LenVsI-1) ) +            # no incoming edge at start vertex
    P * (sym.Sum( X[VtI[i]], (i, 0, LenVtI-1) ) - 1 )**2 +  # exactly one incoming edge at target vertex
    P * sym.Sum( X[VtO[i]], (i, 0, LenVtO-1) ) +            # no incoming edge at target vertex
    P * sym.Sum(                                            # one or zero incoming edges at each central point
        sym.Sum(  X[VI[v,i]] , (i, 0, LenVI[v]-1) ) *
        (sym.Sum(  X[VI[v,i]] , (i, 0, LenVI[v]-1) ) -1) ,
        (v, 0, LenV-1) ) +
    P * sym.Sum(                                            # one or zero outgoing edges at each central point
        sym.Sum(  X[VO[v,i]] , (i, 0, LenVO[v]-1) ) *
        (sym.Sum(  X[VO[v,i]] , (i, 0, LenVO[v]-1) ) -1) ,
        (v, 0, LenV-1) ) +
    P * sym.Sum(                                            # equal number of incoming and outgoing edges at each central point
        (sym.Sum(  X[VI[v,i]] , (i, 0, LenVI[v]-1) ) -
        sym.Sum(  X[VO[v,i]] , (i, 0, LenVO[v]-1) ))**2 ,
        (v, 0, LenV-1) ) )


###################################
###### END PROBLEM DEFINITION #####
###################################



#######################
###### DATA INPUT #####
#######################

import sys, json

data = json.load(sys.stdin)

TestN = data['N']
TestVsI = data['VsI']
TestVsO = data['VsO']
TestVtI = data['VtI']
TestVtO = data['VtO']
TestVI = data['VI']
TestVO = data['VO']
TestC = data['C']
TestP = data['P']

###########################
###### END DATA INPUT #####
###########################



############################
###### DATA PROCESSING #####
############################

## translation of graph structure into dictionaries for sympy

single_valued_dict = {
    N: TestN,
    P: TestP,
    LenVsI: len(TestVsI),
    LenVsO: len(TestVsO),
    LenVtI: len(TestVtI),
    LenVtO: len(TestVtO),
    LenV:  len(TestVI)
}

# multi-valued dictionaries
dict_VsI = { VsI[k]: TestVsI[k] for k in range(len(TestVsI)) }

dict_VsO = { VsO[k]: TestVsO[k] for k in range(len(TestVsO)) }

dict_VtI = { VtI[k]: TestVtI[k] for k in range(len(TestVtI)) }

dict_VtO = { VtO[k]: TestVtO[k] for k in range(len(TestVtO)) }

dict_LenVI = { LenVI[k]: len(TestVI[k]) for k in range(len(TestVI)) }

dict_LenVO = { LenVO[k]: len(TestVO[k]) for k in range(len(TestVO)) }

dict_VI = { VI[k, i]: TestVI[k][i] for k in range(len(TestVI)) for i in range(len(TestVI[k])) }

dict_VO = { VO[k, i]: TestVO[k][i] for k in range(len(TestVO)) for i in range(len(TestVO[k])) }

num_qubits = TestN

dict_costs = { C[k]: TestC[k] for k in range(len(TestC)) }

# definition of cost polynomial
cost_poly = sym.Poly(cost_function
                     .subs(single_valued_dict)
                     .doit()
                     .subs(dict_VsI)
                     .subs(dict_VsO)
                     .subs(dict_VtI)
                     .subs(dict_VtO)
                     .subs(dict_LenVI)
                     .subs(dict_LenVO)
                     .doit()
                     .subs(dict_VI)
                     .subs(dict_VO)
                     .subs(dict_costs),
                     [X[i] for i in range(num_qubits)])

################################
###### END DATA PROCESSING #####
################################



################################
###### EXECUTION ON D WAVE #####
################################

# generate QUBO-matrix for the given number of Qubits
Qubo = { (i,j) : (cost_poly.coeff_monomial(X[i]**1 * X[j]**1)
                  + ( cost_poly.coeff_monomial(X[i]**1) if (i==j) else 0 ))
        for i in range(num_qubits)
        for j in range(i,num_qubits) }

# execution on D-Wave simulator
from dwave_qbsolv import QBSolv
result_dwave = QBSolv().sample_qubo(Qubo)

#####################################
###### END EXECUTION ON D WAVE #####
#####################################



##########################
###### DATA TRANSFER #####
##########################

# take first result
# better: check for index with lowest energy
samples = list(result_dwave.samples())[0]

# select ids of all selected edges
ids = []
for i in range(len(samples)):
    if (samples[i] == 1):
        ids.append(i)
print(json.dumps( {"ids": ids} ))

##############################
###### END DATA TRANSFER #####
##############################









