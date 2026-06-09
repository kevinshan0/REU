using CairoMakie

# set background color
set_theme!(backgroundcolor=:gray90)

# create figure, axis, and colorbar
f = Figure(size=(800, 500))
### aspect reduces the size of the axis, we should edit the layout directly
### to get our intended visual look
### aspect=width/height
# ax = Axis(f[1, 1], aspect=1)
ax = Axis(f[1, 1], aspect=0.5)
Colorbar(f[1, 2])

# adjust layout
### force column to be length 1.0 relative to the first row, which is indexed by 1
### Aspect(row_index, aspect_of_col_relative_to_row)
colsize!(f.layout, 1, Aspect(1, 1.0))

# Box to visualize cells in a figure
Box(f[1, 1], color=(:red, 0.2), strokewidth=0)

# shrinks/enlargens the figure (gray box) to be the same size as the layout
resize_to_layout!(f)

# 25 axis in figure to show clipping
g = Figure(width=800, height=500)
for i in 1:5, j in 1:5
    Axis(g[i, j], width=150, height=150)
end

### figure will clip without resize_to_layout
resize_to_layout!(g)
g

### mental model:
### the figure contains the layout, the layout perfectly wraps the content
### the layout contain cells, which can visualized using Box()
### objects go into the cells, for example: Axis, Colorbar
### these objects have a defined aspect, which fills up the cell
