using JSON3
using QuantumOptics

include("ir.jl")
include("task.jl")



function convert(task_json::String)
    task = JSON3.read(task_json, Task);
    return task
end

function run(task_json::String)
    task = convert(task_json);
    result = _run(task);
    return JSON3.write(to_dict(result))
end

function _run(task::Task)
    runtime = @elapsed begin
        circ = task.program;
        args = task.args;

        b = SpinBasis(1//2);
        f = FockBasis(args.fock_cutoff);

        _map_pauli = Dict(
            "i" => identityoperator(b),
            "x" => sigmax(b),
            "y" => sigmay(b),
            "z" => sigmaz(b)
        );

        _map_ladder = Dict(
            0 => identityoperator(f),
            -1 => destroy(f),
            +1 => create(f),
        );

        function _map_operator_to_qo(operator::Operator)
            _hs = []
            if !isempty(operator.pauli)
                _h_qreg = [_map_pauli[pauli] for pauli in operator.pauli]
                _hs = vcat(_hs, _h_qreg);
            end
            if !isempty(operator.ladder)
                _h_qmode = [prod([_map_ladder[ladder_op] for ladder_op in mode]) for mode in operator.ladder]
                _hs = vcat(_hs, _h_qmode);
            end
            op = operator.coefficient * tensor(_hs...);
            return op
        end

        function _sum_operators(operators::Vector{Operator})
            return sum([_map_operator_to_qo(operator) for operator in operators])
        end

        function _map_gate_to_qobj(gate::AnalogGate)
            return _sum_operators(gate.hamiltonian)
        end

        function _map_metric(metric::Metric, circ::AnalogCircuit)::Function
            if isa(metric, EntanglementEntropyVN)
                return (t, psi) -> entanglement_entropy_vn(psi, metric.qreg, metric.qmode, circ.n_qreg, circ.n_qmode);
            elseif isa(metric, Expectation)
                return (t, psi) -> expect(_sum_operators(metric.operator), psi)
            else
                println("Not a valid metric type.")
            end
        end

        function _initialize()
            state_qreg = [spinup(b) for qreg in 1:circ.n_qreg];
            state_qmode = [fockstate(f, 0) for qmode in 1:circ.n_qmode];
            psi = tensor(vcat(state_qreg, state_qmode)...);
            return psi
        end

        psi = _initialize();
        fmetrics = Dict{String, Function}(key => _map_metric(metric, circ) for (key, metric) in args.metrics)

        data = DataAnalog(
            state=psi,
            metrics=Dict(key => [] for (key, metric) in args.metrics),
        );

        function fout(t, psi)
            data.state = psi;
            for (key, fmetric) in fmetrics
                val = fmetric(t, psi)
                push!(data.metrics[key], val);
            end
        end
        
        t0 = 0.0
        for statement in circ.sequence
            if statement.key == "initialize"
                continue  # todo
            elseif statement.key == "evolve"
                tspan = range(0, stop=statement.gate.duration, step=args.dt);
                append!(data.times, collect(tspan) .+ t0);
                t0 = statement.gate.duration
                
                H = _map_gate_to_qobj(statement.gate);
                timeevolution.schroedinger(tspan, data.state, H; fout=fout);
            elseif statement.key == "measure"
                continue  # todo
            end
        end
    end

    result = TaskResultAnalog(
        times=data.times,
        runtime=runtime,
        state=map(complexf64_to_complexfloat, data.state.data),
        metrics=data.metrics,
    )

    return result
end



function entanglement_entropy_vn(psi, qreg, qmode, n_qreg, n_qmode)
    # note: element-wise +1 accounts for different starting indices between openQSIM/Python & Julia
    rho = ptrace(psi, vcat(qreg .+ 1, qmode .+ n_qreg .+ 1)) 
    return entropy_vn(rho)
end

