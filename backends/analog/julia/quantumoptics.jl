using JSON
using QuantumOptics

include("ir.jl")
include("task.jl")



function convert(task_json::String)
    d = JSON.parse(task_json);
    task = from_dict(Task, d);
    return task
end

function run(task_json::String)
    task = convert(task_json);
    result = evolve(task);
    return result
end

function evolve(task::Task)
    runtime = @elapsed begin
        circ = task.program;
        args = task.args;

        b = SpinBasis(1//2);
        f = FockBasis(args.fock_cutoff);

        _map_qreg = Dict(
            "i" => identityoperator(b),
            "x" => sigmax(b),
            "y" => sigmay(b),
            "z" => sigmaz(b)
        );

        _map_qmode = Dict(
            0 => identityoperator(f),
            -1 => destroy(f),
            +1 => create(f),
        );


        # todo: change to Metrics, rather than operator
        function _map_operator_to_qo(operator::Operator)
            _hs = []
            if !isempty(operator.qreg)
                _h_qreg = [_map_qreg[qreg] for qreg in operator.qreg]
                _hs = vcat(_hs, _h_qreg);
            end
            if !isempty(operator.qreg)
                _h_qmode = [prod([_map_qmode[qmode_op] for qmode_op in mode]) for mode in operator.qmode]
                _hs = vcat(_hs, _h_qmode);
            end
            op = operator.coefficient * tensor(_hs...)
            return op
        end

        function _map_metric(metric::Metric, circuit::AnalogCircuit)
            
        end

        function _initialize()
            state_qreg = [spinup(b) for qreg in 1:circ.n_qreg];
            state_qmode = [fockstate(f, 0) for qmode in 1:circ.n_qmode];
            psi = tensor(vcat(state_qreg, state_qmode)...);
            return psi
        end

        psi = _initialize();

        fmetrics = 

        data = DataAnalog(
            state=psi,
            expect=Dict(name => [] for (name, op) in args.observables)
            metrics=Dict(name => [] for (name))
        );

#         println("Intial state:   ", data.state)

        exp_observable = Dict(
            name => _map_operator_to_qo(op) for (name, op) in args.observables
        )

        function fout(t, psi)
            data.state = psi;
            push!(data.times, t);
            for (name, op) in exp_observable
                push!(data.expect[name], expect(op, psi));
            end
        end

        for gate in circ.sequence
            tspan = range(0, stop=gate.duration, step=args.dt);
            H = sum([_map_operator_to_qo(operator) for operator in gate.unitary]);
            timeevolution.schroedinger(tspan, psi, H; fout=fout);
        end
    end

    result = TaskResultAnalog(
        # expect=data.expect,
        times=data.times,
        runtime=runtime,
        state=data.state.data,
        metrics=data.metrics,
    )

#     println(result);
    return JSON.json(to_dict(result))
end




function entanglement_entropy_vn(t, psi, qreg, qmode, n_qreg, n_qmode)
    rho = ptrace(psi, qreg + [n_qreg + m for m in qmode])
    return entropy_vn(rho)
end

