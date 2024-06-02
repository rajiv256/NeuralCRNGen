import Pkg;
# Pkg.add("Plots")

using Plots;
using ColorSchemes;
using LaTeXStrings;
theme(:default);
palette(:Dark2_5);
# plot(x, y1, label="sin(x)", lw=2, marker=:circle, markersize=4, title="Enhanced Sine Wave", xlabel="x-axis", ylabel="y-axis", legend=:best, grid=false, framestyle=:semi, widen=false)
KWARGS = (
    "lw" => 2,
    "marker" => :circle,
    "markersize" => 4,
    "fontfamily" => "helvetica",
    "grid" => false,
    "framestyle" => :semi,
    "widen" => false,
)

function myplot(xs, ys, labels; xlabel="", ylabel="", marker=:circle, markersize=1, title="", name="plotname", output_dir="", fontsize=18, size=(600, 400), xticks=nothing, yticks=nothing)
    for (x, y, label) in zip(xs, ys, labels)
        plot!(x, y,
            lw=2, marker=marker, markersize=markersize,
            label=label, xlabel=xlabel, ylabel=ylabel,
            size=size,
            legend=:topright, grid=false, framestyle=:semi, widen=false,
            xtickfontsize=fontsize, ytickfontsize=fontsize,
            xguidefontsize=fontsize, yguidefontsize=fontsize,
            legendfontsize=fontsize,
            fontfamily="Arial")
    end
    savefig("./julia/$output_dir/images/$name.svg")
    savefig("./julia/$output_dir/images/$name.png")
end

function myscatter3d(xs, ys, zs, colors, markers; xlabel="", ylabel="", markersize=4, title="", name="plotname", output_dir="", fontsize=18, size=(600, 400))
    # for (x, y, z, c label) in zip(xs, ys, labels)
    scatter3d!(xs, ys, zs, group=colors, markers=markers,
        markersize=markersize, legend=:topright, grid=false,
        framestyle=:semi, widen=false,
        size=size, tickfontsize=fontsize,
        xtickfontsize=fontsize, ytickfontsize=fontsize,
        xguidefontsize=fontsize, yguidefontsize=fontsize,
        legendfontsize=fontsize,
        fontfamily="Arial")
        plot!([], [], xticks=[], yticks=[], zticks=[], label="")
    
    savefig("./julia/$output_dir/images/$name.svg")
    savefig("./julia/$output_dir/images/$name.png")
end

function myscatter(xs, ys, colors; xlabel="", ylabel="", markersize=4, title="", name="plotname", output_dir="", fontsize=18, size=(600, 400))
    gg = scatter!(xs, ys, group=colors,
        markersize=markersize,
        xlabel=xlabel, ylabel=ylabel,
        size=size,
        legend=:topright, grid=false, framestyle=:semi, widen=false,
        xtickfontsize=fontsize, ytickfontsize=fontsize,
        xguidefontsize=fontsize, yguidefontsize=fontsize,
        legendfontsize=fontsize, fontfamily="Arial")
    
    savefig("./julia/$output_dir/images/$name.svg")
    savefig("./julia/$output_dir/images/$name.png")
    return gg
end

function myscatternogroup(xs, ys; xlabel="", ylabel="", markershape=:xcross, markercolor="black", markersize=4, label="errors",
    title="", name="", output_dir="", fontsize=18, size=(600, 400))
    gg = scatter!(xs, ys,
        markersize=markersize,
        markershape=markershape,
        markercolor=markercolor,
        label=label,
        size=size,
        xlabel=xlabel, ylabel=ylabel,
        legend=:topright, grid=false, framestyle=:semi, widen=false,
        xtickfontsize=fontsize, ytickfontsize=fontsize,
        xguidefontsize=fontsize, yguidefontsize=fontsize,
        legendfontsize=fontsize, fontfamily="Arial")
    savefig("./julia/$output_dir/images/$name.svg")
    savefig("./julia/$output_dir/images/$name.png")
    
    return gg
end

# x = [1, 2, 3]
# y = [2.5, 6.6667, 3.5]
# y2 = y .^ 2
# myplot([x, x], [y, y2], ["acwr", "kdff"], xlabel="x", ylabel="y", output_dir="rings", name="test")