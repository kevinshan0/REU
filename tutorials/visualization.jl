using GLMakie
GLMakie.activate!()
using QuantumSavory
using QuantumSavory.ProtocolZoo
using QuantumSavory.StatesZoo
using ConcurrentSim
using Graphs

# Parameters
NODES = 10
T2 = 100.0

# graph
graph = SimpleGraph(NODES)
for i in 1:NODES-1
    add_edge!(graph, NODES, i)
end

# registers
registers = Register[]
for _ in 1:NODES
    push!(registers, Register(rand(1:5), T2Dephasing(T2)))
end

net = RegisterNet(graph, registers)

# simulation
sim = Simulation()

# entangler
for (; src, dst) in edges(net)
    entangler_protocol = EntanglerProt(sim, net, src, dst; rounds=-1)
    @process entangler_protocol()
end

# swapper
for node in vertices(net)
    swapper_protocol

# visualization
figure = Figure()
_, ax, _, obs = registernetplot_axis(figure[1, 1], net)
resize_to_layout!(figure)

step_ts = range(0, 20, step=0.1)
record(figure, "visualization.mp4", step_ts, framerate=10, visible=true) do t
    run(sim, t)
    notify(obs)
    ax.title = "t=$(t)"
end