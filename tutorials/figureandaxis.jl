using CairoMakie

# Data
seconds = 0:0.1:2
measurements = [8.2, 8.4, 6.3, 9.5, 9.1, 10.5, 8.6, 8.2, 10.5, 8.5, 7.2,
    8.8, 9.7, 10.8, 12.5, 11.6, 12.1, 12.1, 15.1, 14.7, 13.1]

f = Figure()
axis = Axis(
    f[1, 1],
    title="axis and figures dummy data",
    xlabel="Time (s)",
    ylabel="Value",
)
scatter!(
    axis,
    seconds,
    measurements,
    color=:tomato,
    label="Measurements"
)
lines!(
    axis,
    seconds,
    exp.(seconds) .+ 7,
    color=:purple,
    linestyle=:dash,
    label="f(x) = exp(x) + 7"
)
axislegend(position=:rb)

save("first_figure.png", f)
save("first_figure.svg", f)
save("first_figure.pdf", f)
