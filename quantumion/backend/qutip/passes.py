from quantumion.backend.qutip.conversion import (
    QutipBackendCompiler,
    QutipExperimentInterpreter
)
from quantumion.compilerv2.walk import Walk, Post
from quantumion.compilerv2.rule import ConversionRule
from quantumion.compilerv2.rewriter import Chain

def compiler_analog_circuit_to_qutipIR(model, fock_cutoff):
    return Post(QutipBackendCompiler(fock_cutoff=fock_cutoff))(model=model)

def compiler_analog_args_to_qutipIR(model, fock_cutoff):
    return Post(QutipBackendCompiler(fock_cutoff=fock_cutoff))(model=model)

def run_qutip_experiment(model):
    return Post(QutipExperimentInterpreter())(model=model)