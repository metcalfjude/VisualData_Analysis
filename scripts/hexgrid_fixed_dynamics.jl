using FractionalSaccades
using KernelDensity
FractionalSaccades.@preamble

file = jldopen(datadir("native", "hexgrid", "gaze.jld2"), "r")
index = file["index"]
HEXCONDS = 1:12

begin # * Set up figure
    f = Figure()
end
begin # * Load the data, chunked by hex grid size
    xs = [only(unique(r.cond_id)) => ToolsArray(getindex.([file], r.gaze_file),
                                                (Dim{:uid}(r.uid),))
          for r in DataFrames.groupby(index, :cond_id)]
    xs = ToolsArray(last.(xs), (Dim{:condition}(first.(xs)),))
    gaze = filter.(x -> !all(isnan.(x)), xs)
end
# --------------- Fixation period: distribution, msd, and autocorrelation --------------- #
begin # * Distribution
    fix = progressmap(gaze; parallel = true) do x
        d = map(x) do y
            y = y[𝑡(0 .. DimensionalData.metadata(y)["fix_off"])]
        end
    end
end
begin
    pdfs = map(fix) do x
        x = cat(x...; dims = 𝑡())
        x = filter(!isnan, x)
        k = kde(filter(!isnan, x); boundary = (-25, 25))
        ToolsArray(k.density, (Dim{:position}(k.x),))
    end
    pdfs = stack(pdfs, dims = 2)
end
# begin
#     pdfs = progressmap(fix; parallel = true) do x
#         d = map(x) do y
#             m = map(eachslice(y; dims = (2, 3))) do z
#                 k = kde(filter(!isnan, z); boundary = (-25, 25))
#                 ToolsArray(k.density, (Dim{:position}(k.x),))
#             end
#         end
#         stack(d, dims = 1)
#     end
# end
begin
    f = Figure()
    ax = Axis(f[1, 1])
    colormap = cgrad(darksunset, length(HEXCONDS), categorical = true)
    # for (_i, i) in enumerate(sort(HEXCONDS))
    #     p = pdfs[Dim{:condition}(At(i))]
    #     plot!(ax, lookup(p, 1), parent(p) .+ _i * 0.01; color = colormap[_i])
    # end
    p = stackedtraces!(ax, pdfs[Dim{:condition}(At(HEXCONDS))]; offset = 1.1, colormap,
                       linewidth = 4, spacing = :even)
    Colorbar(f[1, 2]; colormap, label = "Hex grid size",
             colorrange = (minimum(HEXCONDS) - 0.5, maximum(HEXCONDS) + 0.5),
             ticks = HEXCONDS)
    f
end
close(file)
