import Pkg;
# Pkg.add("Plots")

using Plots;
using ColorSchemes;
using LaTeXStrings;
theme(:default);
palette(:Dark2_5);
# plot(x, y1, label="sin(x)", lw=2, marker=:circle, markersize=4, title="Enhanced Sine Wave", xlabel="x-axis", ylabel="y-axis", legend=:best, grid=false, framestyle=:semi, widen=false)
# KWARGS = (
#     "lw" => 2,
#     "marker" => :circle,
#     "markersize" => 4,
#     "fontfamily" => "helvetica",
#     "grid" => false,
#     "framestyle" => :semi,
#     "widen" => false,
# )

# Set global font sizes
default(
    legendfontsize=18,            # legend font size
    tickfontsize=16,              # tick font size
    guidefontsize=18,             # axis label font size
    titlefontsize=14,             # title font size
    # Optional: you can also set font family
    fontfamily="Arial",  # or "Helvetica", "Times New Roman", etc.,
    # Option 3: Using hex colors
    palette=["#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377", 
           "#66CCEE", "#BBBBBB", "#EE3377", "#000000", "#CC6677",], 
    markersize=2,
    framestyle=:semi,
    # widen=false,
    lw=2, 
    size=(600, 450),  # Wider figure
    bottom_margin=3Plots.mm, # Add global margin
    legend=:best
)


function myplot(xs, ys, labels; xlabel="", ylabel="", marker=:circle, markersize=4, title="", name="plotname", output_dir="", fontsize=18, size=(600, 400), linestyle=:solid)
    for (x, y, label) in zip(xs, ys, labels)
        plot!(x, y,
            lw=2, marker=marker, markersize=markersize,
            label=label, xlabel=xlabel, ylabel=ylabel, linestyle=linestyle)
    end
    savefig("./julia/$output_dir/images/$name.svg")
    savefig("./julia/$output_dir/images/$name.png")
end

function myscatter3d(xs, ys, zs, colors, markers; xlabel="", ylabel="", markersize=4, title="", name="plotname", output_dir="", fontsize=18, size=(600, 400))
    # for (x, y, z, c label) in zip(xs, ys, labels)
    scatter3d!(xs, ys, zs, group=colors, markers=markers,
        markersize=markersize, legend=:topright, grid=false)
    savefig("./julia/$output_dir/images/$name.svg")
    savefig("./julia/$output_dir/images/$name.png")
end

function myscatter(xs, ys, colors; xlabel="", ylabel="", markersize=4, markershape=:circle, title="", name="plotname", output_dir="", fontsize=18, size=(600, 400))
    gg = scatter!(xs, ys, group=colors,
        markersize=markersize, markershape=markershape,
        xlabel=xlabel, ylabel=ylabel)
    savefig("./julia/$output_dir/images/$name.svg")
    savefig("./julia/$output_dir/images/$name.png")
    return gg
end

function myscatternogroup(xs, ys; xlabel="", ylabel="", markershape=:xcross, markersize=4, label="errors", 
    title="", name="", output_dir="", fontsize=18, size=(600, 400))
    gg = scatter!(xs, ys,
        markersize=markersize,
        markershape=markershape,
        label=label,
        xlabel=xlabel, ylabel=ylabel)
    savefig("./julia/$output_dir/images/$name.svg")
    savefig("./julia/$output_dir/images/$name.png")
    return gg
end


function plot_regression_dataset(train, mini, maxi, func; output_dir="")
    plot()
    X= range(mini-0.5, maxi+0.5, length=100)
    Y = range(mini-0.5, maxi+0.5, length=100)
    Z = [func(x1, x2) for x1 in X, x2 in Y]
    
    g = surface(X, Y, Z,
     alpha=0.8,
     color=:blues,
     colorbar=false)
    # g = plot_mesh_grid(mini, maxi, func, train)
    

    scatter3d!(g, getindex.(train, 1), getindex.(train, 2), getindex.(train, 3), xlabel=L"$x_1$", ylabel=L"$x_2$", zlabel=L"$y$", markersize=4, legend=false)

    savefig(g, "julia/$output_dir/regplot.png")
    savefig(g, "julia/$output_dir/regplot.svg")
end


using Plots

# Define the function to calculate z
function plot_mesh_grid(mini, maxi, func, train)
    
    g = surface(getindex.(train, 1), getindex.(train, 2), getindex.(train, 3), xlabel="x1", ylabel="x2", zlabel="z", color=:blues, alpha=0.4, colorbar=false)
    

    return g
end
