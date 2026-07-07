using ConcurrentSim
using Distributions
using Plots
using Random
using ResumableFunctions
using Statistics

# # A minimal repeater
#
# This example keeps the discrete-event structure visible:
# - two independent link generators create elementary entanglement,
# - a repeater waits until it has one left-link and one right-link pair,
# - the repeater swaps them into one end-to-end pair.
#
# The `Store`s play the role of the repeater memories.  A link generator blocks when
# its memory is full, and the repeater blocks when one side has no pair available.

abstract type MemoryContent end
struct EmtpyMemory <: MemoryContent end
struct LinkEntanglement <: MemoryContent end

is_entanglement(content::MemoryContent) = false
is_entanglement(content::LinkEntanglement) = true

get_left(station::RepeaterStation) = get(station.left_memory, is_entanglement)
get_right(station::RepeaterStation) = get(station.right_memory, is_entanglement)
put_left!(station::RepeaterStation, content::MemoryContent) = put!(station.left_memory, content)
put_right!(station::RepeaterStation, content::MemoryContent) = put!(station.right_memory, content)

mutable struct RepeaterStation
    left_memory::Store{MemoryContent}
    right_memory::Store{MemoryContent}
    # how many e2e pairs have been completed?
    completed_pairs::Int
    # At what times were the e2e pairs completed?
    completion_times::Vector{Float64}
end

function RepeaterStation(env::Environment; memory_size=1)
    RepeaterStation(
        Store{MemoryContent}(env; capacity=memory_size),
        Store{MemoryContent}(env; capacity=memory_size),
        0,
        Float64[],
    )
end

@resumable function entangler(
        env::Environment,
        memory::Store{MemoryContent},
        success_probability::Float64,
        attempt_time::Float64,
    )
    # attempt entanglement until success, repeat forever.
    ###############
    # Code here
    ###############
    while true
        put!(memory, EmtpyMemory())
        @yield get(memory)
        @yield timeout(env, attempt_time)
        if rand() > success_probability
            continue
        end
        put!(memory, LinkEntanglement())
    end
end

@resumable function repeater(
        env::Environment,
        station::RepeaterStation,
        swap_time::Float64,
    )
    # wait for a left and a right entanglement, then swap them into an end-to-end pair. Repeat forever.
    ###############
    # Code here
    ###############
    while true
        left = get(station.left_memory, isequal(LinkEntanglement()))
        right = get(station.right_memory, isequal(LinkEntanglement()))
        @yield left & right
        @yield timeout(env, swap_time)
        put!(station.left_memory, EmtpyMemory())
        put!(station.right_memory, EmtpyMemory())
        completed_pairs += 1
        push!(completion_times, now(env))
    end
end

function run_repeater(;
        simulation_time=1_000.0,
        left_success_probability=0.2,
        right_success_probability=0.2,
        attempt_time=1.0,
        swap_time=1.0,
        memory_size=1,
        seed=1234,
    )
    Random.seed!(seed)

    sim = Simulation()
    station = RepeaterStation(sim; memory_size)

    @process entangler(
        sim, station.left_memory, left_success_probability, attempt_time)
    @process entangler(
        sim, station.right_memory, right_success_probability, attempt_time)
    @process repeater(sim, station, swap_time)

    run(sim, simulation_time)
    station
end

function sampled_counts(completion_times, sample_times)
    counts = zeros(Int, length(sample_times))
    completed = 0

    for (i, sample_time) in pairs(sample_times)
        while completed < length(completion_times) &&
                completion_times[completed + 1] <= sample_time
            completed += 1
        end
        counts[i] = completed
    end

    counts
end

cycle_times(station::RepeaterStation) = diff([0.0; station.completion_times])

function plot_repeater_run(station::RepeaterStation; simulation_time=1_000.0)
    sample_times = range(0, simulation_time; length=200)
    counts = sampled_counts(station.completion_times, sample_times)

    throughput_plot = plot(
        sample_times,
        counts;
        xlabel="Simulation time",
        ylabel="Completed end-to-end pairs",
        label=false,
        linewidth=2,
        title="Repeater throughput",
    )

    cycle_plot = histogram(
        cycle_times(station);
        xlabel="Time between end-to-end pairs",
        ylabel="Count",
        label=false,
        bins=30,
        title="Completion-time fluctuations",
    )

    plot(throughput_plot, cycle_plot; layout=(1, 2), size=(900, 350))
end

simulation_time = 1_000.0
station = run_repeater(; simulation_time)

@assert station.completed_pairs == length(station.completion_times)
@assert issorted(station.completion_times)
@assert all(diff(station.completion_times) .> 0)

println("Completed pairs: $(station.completed_pairs)")
println("Mean time between pairs: $(round(mean(cycle_times(station)), sigdigits=4))")

display(plot_repeater_run(station; simulation_time))
