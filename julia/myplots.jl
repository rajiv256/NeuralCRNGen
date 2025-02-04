import Pkg;
# Pkg.add("Plots")

using Plots;
using ColorSchemes;
using LaTeXStrings;
theme(:default);
palette(:Dark2_5);


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


# x = [1, 2, 3]
# y = [2.5, 6.6667, 3.5]
# y2 = y .^ 2
# myplot([x, x], [y, y2], ["acwr", "kdff"], xlabel="x", ylabel="y", output_dir="rings", name="test")