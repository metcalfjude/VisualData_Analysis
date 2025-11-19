using FractionalSaccades
using KernelDensity
using Autocorrelations
import TimeseriesTools.msdist
FractionalSaccades.@preamble
using Foresight
set_theme!(foresight(:physics))

filename = datadir("native", "hexgrid", "gaze.jld2")
HEXCONDS = 1:12
SCATTERCONDS = 13:16
FREECONDS = 17:36
NMIN = 2000
FIXATION(m) = 0 .. m["fix_off"]
SEARCH(m) = m["fix_off"] .. m["tar_on"]
PERIOD = SEARCH
# PERIOD = FIXATION

index, gaze = jldopen(filename, "r") do file # * Load the data, chunked by hex grid size
    index = file["index"]
    xs = [only(unique(r.cond_id)) => ToolsArray(getindex.([file], r.gaze_file),
                                                (Dim{:uid}(r.uid),))
          for r in DataFrames.groupby(index, :cond_id)]
    xs = ToolsArray(last.(xs), (Dim{:condition}(first.(xs)),))
    gaze = filter.(x -> !all(isnan.(x)), xs)
    return (index, gaze)
end
begin # * Plot sample path on image
    map(2:2:36) do c
        map(1:10) do i
            img = load(datadir("native", "hexgrid", "gallery", "$c.jpg"))
            x = gaze[c][i][Eye = At(:right)]
            y = x[:, 2] .+ 1200 / 2
            x = x[:, 1] .+ 1920 / 2

            f = OnePanel()
            ax = Axis(f[1, 1]; aspect = DataAspect())
            hidedecorations!(ax)
            hidespines!(ax)
            image!(ax, rotr90(img))
            # lines!(ax, collect(x), collect(y); color = :white, linewidth = 3)
            lines!(ax, collect(x), collect(y); color = :black, linewidth = 3)
            lines!(ax, collect(x), collect(y); color = eachindex(x), linewidth = 2,
                   colormap = :turbo)
            # display(f)
            wsave(projectdir("plots", "sample_paths",
                             "condition_$c", "index_$i.pdf"), f)
            return nothing
        end
    end
end
