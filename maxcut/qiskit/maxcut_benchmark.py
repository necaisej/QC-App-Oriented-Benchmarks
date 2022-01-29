"""
MaxCut Benchmark Program - Qiskit
"""

import sys
import time
from collections import namedtuple

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute # for computing expectation tables

sys.path[1:1] = [ "_common", "_common/qiskit" ]
sys.path[1:1] = [ "../../_common", "../../_common/qiskit" ]
import execute as ex
import metrics as metrics

import common       # from lanl-ansi-max-cut

np.random.seed(0)

verbose = False

# Variable for number of resets to perform after mid circuit measurements
num_resets = 1

# saved circuits for display
QC_ = None
Uf_ = None

# based on examples from https://qiskit.org/textbook/ch-applications/qaoa.html
QAOA_Parameter  = namedtuple('QAOA_Parameter', ['beta', 'gamma'])


############### Circuit Definition
  
# Create ansatz specific to this problem, defined by G = nodes, edges, and the given parameters
# Do not include the measure operation, so we can pre-compute statevector
def create_qaoa_circ(nqubits, edges, parameters):

    qc = QuantumCircuit(nqubits)

    # initial_state
    for i in range(0, nqubits):
        qc.h(i)

    for par in parameters:
        #print(f"... gamma, beta = {par.gamma} {par.beta}")
        
        # problem unitary
        for i,j in edges:
            qc.rzz(2 * par.gamma, i, j)

        qc.barrier()
        
        # mixer unitary
        for i in range(0, nqubits):
            qc.rx(2 * par.beta, i)

    return qc
    
# Create the benchmark program 
def MaxCut (num_qubits, secret_int, edges, method = 1, rounds = 1, theta=None,):
    
    # IF angles dict are given, then bypass method check;
    # just retern the quantum circuit with those angles
    
    if theta is None:
        if method == 1:
            # Set default angles
            theta = 2*rounds*[1.0]
        
    # put parameters into the form expected by the ansatz generator
    p = len(theta)//2  # number of qaoa rounds
    beta = theta[:p]
    gamma = theta[p:]
    parameters = [QAOA_Parameter(*t) for t in zip(beta,gamma)]
        
    # and create the circuit, without measurements
    qc = create_qaoa_circ(num_qubits, edges, parameters)   

    # pre-compute and save an array of expected measurements
    compute_expectation(qc, num_qubits, secret_int)
        
    # add the measure here
    qc.measure_all()
        

    # save small circuit example for display
    global QC_
    if QC_ == None or num_qubits <= 6:
        if num_qubits < 9: QC_ = qc

    # return a handle on the circuit
    return qc


############### Expectation Tables

# DEVNOTE: We are building these tables on-demand for now, but for larger circuits
# this will need to be pre-computed ahead of time and stored in a data file to avoid run-time delays.

# dictionary used to store pre-computed expectations, keyed by num_qubits and secret_string
# these are created at the time the circuit is created, then deleted when results are processed
expectations = {}

# Compute array of expectation values in range 0.0 to 1.0
# Use statevector_simulator to obtain exact expectation
def compute_expectation(qc, num_qubits, secret_int, backend_id='statevector_simulator'):
    
    #execute statevector simulation
    sv_backend = Aer.get_backend(backend_id)
    sv_result = execute(qc, sv_backend).result()

    # get the probability distribution
    counts = sv_result.get_counts()

    #print(f"... statevector expectation = {counts}")
    
    # store in table until circuit execution is complete
    id = f"_{num_qubits}_{secret_int}"
    expectations[id] = counts

    #print(f"  ... time to execute statevector simulator: {time.time() - ts}")
    
# Return expected measurement array scaled to number of shots executed
def get_expectation(num_qubits, secret_int, num_shots):

    # find expectation counts for the given circuit 
    id = f"_{num_qubits}_{secret_int}"
    if id in expectations:
        counts = expectations[id]
        
        # scale to number of shots
        for k, v in counts.items():
            counts[k] = round(v * num_shots)
        
        # delete from the dictionary
        del expectations[id]
        
        return counts
        
    else:
        return None
    
    
############### Result Data Analysis

# Compare the measurement results obtained with the expected measurements to determine fidelity
def analyze_and_print_result (qc, result, num_qubits, secret_int, num_shots):

    # obtain counts from the result object
    counts = result.get_counts(qc)
    
    # retrieve pre-computed expectation values for the circuit that just completed
    expected_dist = get_expectation(num_qubits, secret_int, num_shots)
    
    if verbose: print(f"For width {num_qubits} problem {secret_int}\n  measured: {counts}\n  expected: {expected_dist}")

    # use our polarization fidelity rescaling
    fidelity = metrics.polarization_fidelity(counts, expected_dist)

    if verbose: print(f"For secret int {secret_int} fidelity: {fidelity}")
    
    return counts, fidelity

# Compute the objective function on a given sample
def compute_objective(results, nodes, edges):
    counts = results.get_counts()
    
    avg = 0
    sum_count = 0
    for solution, count in counts.items():
        obj = -1*common.eval_cut(nodes, edges, solution)

        avg += obj * count
        sum_count += count

    return avg/sum_count


################ Benchmark Loop

# Problem definitions only available for up to 10 qubits currently
MAX_QUBITS = 10
saved_result = None
instance_filename = None

# Execute program with default parameters
def run (min_qubits=3, max_qubits=6, max_circuits=3, num_shots=100,
        method=1, rounds=1,
        backend_id='qasm_simulator', provider_backend=None,
        hub="ibm-q", group="open", project="main", exec_options=None):
    global circuits_done
    
    print("MaxCut Benchmark Program - Qiskit")

    # validate parameters (smallest circuit is 4 qubits)
    max_qubits = max(4, max_qubits)
    max_qubits = min(MAX_QUBITS, max_qubits)
    min_qubits = min(max(4, min_qubits), max_qubits)
    max_circuits = min(10, max_circuits)
    #print(f"min, max qubits = {min_qubits} {max_qubits}")
    
    # Initialize metrics module
    metrics.init_metrics()

    # Define custom result handler
    def execution_handler (qc, result, num_qubits, s_int, num_shots):  
     
        # determine fidelity of result set
        num_qubits = int(num_qubits)
        counts, fidelity = analyze_and_print_result(qc, result, num_qubits, int(s_int), num_shots)
        metrics.store_metric(num_qubits, s_int, 'fidelity', fidelity)

    def execution_handler2 (qc, result, num_qubits, s_int, num_shots):
        global saved_result
        global instance_filename
        
        nodes, edges = common.read_maxcut_instance(instance_filename)
        opt, _ = common.read_maxcut_solution(instance_filename)
        
        f = -1 * compute_objective(result, nodes, edges) / opt
        metrics.store_metric(num_qubits, s_int, 'fidelity', f)
        
        saved_result = result
     
    # Initialize execution module using the execution result handler above and specified backend_id
    if method == 2:
        ex.max_shots = 1
        ex.init_execution(execution_handler2)
    else:
        ex.init_execution(execution_handler)
    
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
            hub=hub, group=group, project=project, exec_options=exec_options)

    # for noiseless simulation, set noise model to be None
    # ex.set_noise_model(None)

    # Execute Benchmark Program N times for multiple circuit sizes
    # Accumulate metrics asynchronously as circuits complete
    # DEVNOTE: increment by 2 to match the collection of problems in 'instance' folder
    for p_depth in [2, 4]:
        for num_qubits in range(min_qubits, max_qubits + 1, 2):
            
            # determine number of circuits to execute for this group
            #num_circuits = min(2**(num_qubits), max_circuits)
            num_circuits = max_circuits
        
            print(f"************\nExecuting [{num_circuits}] circuits with num_qubits = {num_qubits}")
          
            # loop over limited # of inputs for this
            # assume the solution files start with 3 and go up from there
            circuits_done = 0
            for i in range(3, 3 + max_circuits):
            
            
            
                # create integer that represents the problem instance; use s_int as circuit id
                s_int = i
                #print(f"  ... i={i} s_int={s_int}")
            
                # create filename from num_qubits and circuit_id (s_int), then load the problem file
                global instance_filename
                instance_filename = f"instance/mc_{str(num_qubits).zfill(3)}_{str(i).zfill(3)}_000.txt"
                #print(f"... instance_filename = {instance_filename}")
                nodes, edges = common.read_maxcut_instance(instance_filename)
                #print(f"nodes = {nodes}")
                #print(f"edges = {edges}")
            
                # if the file does not exist, we are done with this number of qubits
                if nodes == None:
                    print(f"  ... problem {str(i).zfill(3)} not found, limiting to {circuits_done} circuit(s).")
                    break;
            
            
                if method != 2:
            
                    circuits_done += 1
            
                    # create the circuit for given qubit size and secret string, store time metric
                    ts = time.time()
                    qc = MaxCut(num_qubits, s_int, edges, method, rounds)
                    metrics.store_metric(num_qubits, s_int, 'create_time', time.time()-ts)

                    # collapse the sub-circuit levels used in this benchmark (for qiskit)
                    qc2 = qc.decompose()

                    # submit circuit for execution on target (simulator, cloud simulator, or hardware)
                    ex.submit_circuit(qc2, num_qubits, s_int, shots=num_shots)
                
            
                if method == 2:
                
                    thetas_init = 2*p_depth*[1.0]
                
                    def expectation(theta):
                        global circuits_done
                    
                        circuits_done += 1
                    
                        # Every circuit needs a unique id; add circuits_done instead of s_int
                        unique_id = s_int*1000 + circuits_done
                    
                        # create the circuit for given qubit size and secret string, store time metric
                        ts = time.time()
                        qc = MaxCut(num_qubits, unique_id, edges, method, rounds, theta)
                        metrics.store_metric(num_qubits, unique_id, 'create_time', time.time()-ts)

                        # collapse the sub-circuit levels used in this benchmark (for qiskit)
                        qc2 = qc.decompose()

                        # submit circuit for execution on target (simulator, cloud simulator, or hardware)
                        ex.submit_circuit(qc2, num_qubits, unique_id, shots=num_shots)
                    
                        # Must wait for circuit to complete
                        #ex.throttle_execution(metrics.finalize_group)
                        ex.finalize_execution(metrics.finalize_group)
                    
                        return compute_objective(saved_result, nodes, edges)
                
                    res = minimize(expectation, thetas_init, method='COBYLA')
                    opt, sol = common.read_maxcut_solution(instance_filename)
                
                    num_qubits = int(num_qubits)
                    #counts, fidelity = analyze_and_print_result(qc, result, num_qubits, int(s_int), num_shots)
                    fidelity = -1 * res.fun / opt #known optimum
                
                    metrics.store_metric(num_qubits, s_int, 'fidelity', fidelity)
                    metrics.store_metric(num_qubits, s_int, 'rounds', p_depth)
                
                    #print(res)
                
                
        # Wait for some active circuits to complete; report metrics when groups complete
        ex.throttle_execution(metrics.finalize_group)
        
    # Wait for all active circuits to complete; report metrics when groups complete
    ex.finalize_execution(metrics.finalize_group)

    # print a sample circuit
    print("Sample Circuit:"); print(QC_ if QC_ != None else "  ... too large!")
    #if method == 1: print("\nQuantum Oracle 'Uf' ="); print(Uf_ if Uf_ != None else " ... too large!")

    # Plot metrics for all circuit sizes
    metrics.plot_metrics(f"Benchmark Results - MaxCut ({method}) - Qiskit")

# if main, execute method
if __name__ == '__main__': run(method=2)
   
