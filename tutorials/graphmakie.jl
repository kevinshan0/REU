using Graphs
using CairoMakie

function createGraph()
    while true
        graph = erdos_renyi(50, 0.3)
        if is_connected(graph)
            return graph
        end
    end
end

graph = createGraph()




