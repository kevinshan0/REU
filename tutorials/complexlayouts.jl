using CairoMakie
using FileIO

f = Figure(
    backgroundcolor=RGBf(0.98, 0.98, 0.98),
    size=(1000, 700)
)

# define drig layout
### A and B are in the same column, C and D are in the same column
### but rows cannot be cleanly defined across both columns
gridA = f[1, 1] = GridLayout()
gridB = f[2, 1] = GridLayout()
### the f[1:2, 2] syntax means gridcolCD contains 2 rows (1 through 2) in the second column
### f[1:5, 3] would means 5 rows (1 through 5) in the third column
gridcolCD = f[1:2, 2] = GridLayout()
### this syntax means gridcolCD is a 2 by 1 matrix
gridC = gridcolCD[1, 1]
gridD = gridcolCD[2, 1]

# panel A
axtop = Axis(gridA[1, 1])
axmain = Axis(
    gridA[2, 1],
    xlabel="before",
    ylabel="after"
)
axright = Axis(gridA[2, 2])

# link axes
linkxaxes!(axmain, axtop)
linkyaxes!(axmain, axright)

labels = ["treatment", "placebo", "control"]
### randn(a, b, ..., n) creates an n-dimensional array with size a x b x ... n
### easier to think:
### randn(3, 100, 2) creates 2 matrices both of size 3 by 100
### the .+ [1, 3, 5] indicates each column in each 3 x 100 matrix adds [1, 3, 5].
data = randn(3, 100, 2) .+ [1, 3, 5]

for (label, col) in zip(labels, eachslice(data, dims=1))
    scatter!(axmain, col, label=label)
    density!(axtop, col[:, 1])
    density!(axright, col[:, 2], direction=:y)
end

# remove gap between data and axes
xlims!(axtop, low=0)
ylims!(axright, low=0)

# change ticks
axmain.xticks = 0:3:9
axtop.xticks = 0:3:9

# legend
legend = Legend(gridA[1, 2], axmain)

# hide decorations on top and righ axes and explicitly define gaps between axes
hidedecorations!(axtop, grid=false)
hidedecorations!(axright, grid=false)
colgap!(gridA, 10)
rowgap!(gridA, 10)

# explicitly tell legend to fill its cell
legend.tellheight = true

# create title by creating label across top two elements
Label(
    gridA[1, 1:2, Top()],
    "Stimulus ratings",
    valign=:bottom,
    font=:bold,
    padding=(0, 0, 5, 0)
)

f