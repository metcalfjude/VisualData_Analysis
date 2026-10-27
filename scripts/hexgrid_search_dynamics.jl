using VisualData_Analysis
using KernelDensity
using Autocorrelations
import TimeseriesTools.msdist
VisualData_Analysis.@preamble
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
begin # * Plot a sample velocity time series
    subx = gaze[8][1]
    subv = centralderiv(subx, dims = 𝑡)
    subv = subv[:, :, 1][𝑡(PERIOD(DimensionalData.metadata(subv)))] # Left eye
    subv = eachcol(subv) |> (xy -> sqrt.(xy[1] .^ 2 .+ xy[2] .^ 2))
    f = Figure()
    ax = Axis(f[1, 1]; xlabel = "Time (s)", ylabel = "Velocity (px/s)")
    lines!(ax, decompose(subv)...; linewidth = 2, color = collect(subv),
           colormap = cgrad([cornflowerblue, crimson], [0.05, 1]; categorical = true))
    f
end
begin # * Filter
    allfix = progressmap(gaze; parallel = true) do x
        d = map(x) do y
            m = DimensionalData.metadata(y)
            y = y[𝑡(PERIOD(m))][25:end, :, :] # Remove transient
            insac = m["saccade_onsets"] .∈ [PERIOD(m)]
            m["saccade_rate"] = sum(insac) / duration(y)
            y = rebuild(y; metadata = m)
        end
        if PERIOD != FIXATION
            d = filter(x -> size(x, 1) > NMIN, d) # Filter time series where the target is at the centre
        end
        d = filter(x -> !any(isnan.(x)), d)
    end
end
begin # * Filter low saccade rates
    fix = allfix
    # fix = map(fix) do x
    #     x = filter(x) do y
    #         m = DimensionalData.metadata(y)
    #         m["saccade_rate"] > 4
    #     end
    #     # map(x) do y
    #     #     y[1:4:end, :, :] # Downsample
    #     # end
    # end
end
# # --------------- Fixation period: distribution, msd, and autocorrelation --------------- #
# begin
#     pdfs = map(fix) do x
#         x = cat(x...; dims = 𝑡())
#         x = filter(!isnan, x)
#         k = kde(filter(!isnan, x); boundary = (-50, 50))
#         ToolsArray(k.density, (Dim{:position}(k.x),))
#     end
#     pdfs = stack(pdfs, dims = 2)
# end
# begin
#     f = Figure()
#     ax = Axis(f[1, 1])
#     colormap = cgrad(darksunset, length(HEXCONDS), categorical = true)
#     p = stackedtraces!(ax, pdfs[Dim{:condition}(At(HEXCONDS))]; offset = 1.1, colormap,
#                        linewidth = 4, spacing = :even)
#     Colorbar(f[1, 2]; colormap, label = "Hex grid size",
#              colorrange = (minimum(HEXCONDS) - 0.5, maximum(HEXCONDS) + 0.5),
#              ticks = HEXCONDS)
#     f
# end

# ------------------------------------------ msd ----------------------------------------- #
begin
    lags = 1:(minimum(size.(Iterators.flatten(fix), 1)) - 2)
    msd = progressmap(fix; parallel = true) do z
        map(z) do x
            m = map(eachslice(x, dims = (2, 3))) do y
                msdist(y[:, 1, 1], lags)
            end
            stack(m)
        end |> stack
    end
    mmsd = dropdims.(mean.(msd, dims = Dim{:uid}); dims = Dim{:uid}) |> stack
end
begin # * Autocorrelation (on radial velocity)
    velocity = progressmap(fix; parallel = true) do z
        map(z) do x
            dropdims(sqrt.(sum(centraldiff(x, dims = 1) .^ 2, dims = Var)), dims = Var)
        end
    end
    acceleration = progressmap(velocity; parallel = true) do z
        map(z) do x
            centraldiff(x, dims = 1)
        end
    end
    ac = progressmap(velocity; parallel = true) do z
        map(z) do x
            m = map(eachslice(x, dims = 2)) do y
                acf(y[:, 1, 1], lags; normalize = true, demean = true)
            end
            stack(m)
        end |> stack
    end

    mac = dropdims.(mean.(ac, dims = Dim{:uid}); dims = Dim{:uid}) |> stack
end

begin # * Autocorrelation of position
    acp = progressmap(fix; parallel = true) do z
        map(z) do x
            m = map(eachslice(x, dims = (2, 3))) do y
                # acf(y[:, 1, 1], lags; normalize = true, demean = true)
                w = autocor(y[:, 1, 1], lags)
                w = Timeseries(lags .* samplingperiod(y), w)
            end
            stack(m)
        end |> stack
    end
    macp = dropdims.(mean.(acp, dims = Dim{:uid}); dims = Dim{:uid}) |> stack
    psd = progressmap(velocity; parallel = true) do z
        map(z) do x
            m = map(eachslice(x, dims = 2)) do y
                powerspectrum(y, 1; padding = 10) # x component only
            end
            stack(m)
        end |> stack
    end
    mpsd = dropdims.(mean.(psd, dims = Dim{:uid}); dims = Dim{:uid}) |> stack
end

begin # * Set up figure
    f = SixPanel()
    gs = subdivide(f, 5, 3)
    conds = ["Hex", "Scatter", "Free"]
    for (i, CONDS) in enumerate([HEXCONDS, SCATTERCONDS, FREECONDS])
        begin
            ax = Axis(gs[i, 1][1, 1]; xscale = log10, yscale = log10,
                      title = string(PERIOD) * ": " * conds[i],
                      ylabel = "Mean MSD", xlabel = "Lag (s)")
            i > 1 && hideydecorations!(ax)
            ax.ygridvisible = true

            M = mmsd[Var = At(:x), Eye = At(:left), condition = At(CONDS)]
            μ = dropdims(mean(M, dims = :condition); dims = :condition)
            σ = dropdims(std(M, dims = :condition); dims = :condition)
            bandwidth!(ax, decompose(μ)..., linewidth = collect(σ), color = (:crimson, 0.6))
            lines!(ax, decompose(μ)...; linewidth = 3, color = :crimson)

            if PERIOD == FIXATION
                fy = μ[𝑡(1e-3 .. 1e-2)]
                fx = collect(times(fy))
                fy = collect(fy)
                bm = [ones(length(fx)) log10.(fx)] \ log10.(fy)
                fy = exp10.(bm[1] .+ bm[2] * log10.(fx))
                lines!(ax, fx, fy; color = :black, linestyle = :dash,
                       label = "α = $(round(bm[2], digits = 2))")

                fy = μ[𝑡(1e-2 .. 1)]
                fx = collect(times(fy))
                fy = collect(fy)
                bm = [ones(length(fx)) log10.(fx)] \ log10.(fy)
                fy = exp10.(bm[1] .+ bm[2] * log10.(fx))
                lines!(ax, fx, fy; color = :blue, linestyle = :dash,
                       label = "α = $(round(bm[2], digits = 2))")
            else
                fy = μ[𝑡(1e-3 .. 1e-1)]
                fx = collect(times(fy))
                fy = collect(fy)
                bm = [ones(length(fx)) log10.(fx)] \ log10.(fy)
                fy = exp10.(bm[1] .+ bm[2] * log10.(fx))
                lines!(ax, fx, fy; color = :black, linestyle = :dash,
                       label = "α = $(round(bm[2], digits = 2))")
            end

            axislegend(ax, position = :rb)
            f
        end
        begin # * Plot
            ax = Axis(gs[i, 4]; xlabel = "Lag (s)", ylabel = "Velocity autocorrelation")
            i > 1 && hideydecorations!(ax)
            ax.ygridvisible = true

            m = mac[Eye = At(:left), condition = At(CONDS)][𝑡(0 .. 0.5)]
            μ = dropdims(mean(m, dims = :condition); dims = :condition)
            σ = dropdims(std(m, dims = :condition); dims = :condition)
            bandwidth!(ax, decompose(μ)..., linewidth = collect(σ), color = (:black, 0.6))
            lines!(ax, decompose(μ)...; linewidth = 3, color = :black)
            f
        end
        begin # * Plot
            ax = Axis(gs[i, 5])
            i > 1 && hideydecorations!(ax)
            ax.ygridvisible = true

            spectrumplot!(ax,
                          mpsd[Eye = At(:left), condition = At(CONDS), Var = At(:x)][2:(end - 2),
                                                                                     :];
                          linewidth = 2)
            ax.ylabel = "Spectral density, v"
            ax.xlabel = "Frequency (Hz)"
            f
        end
        begin
            ax = Axis(gs[i, 3]; xlabel = "Lag (s)", ylabel = "Autocorrelation")
            i > 1 && hideydecorations!(ax)
            ax.ygridvisible = true

            m = macp[Eye = At(:left), condition = At(CONDS), Var = At(:x)]
            μ = dropdims(mean(m, dims = :condition); dims = :condition)
            σ = dropdims(std(m, dims = :condition); dims = :condition)
            bandwidth!(ax, decompose(μ)..., linewidth = collect(σ), color = (:black, 0.6))
            lines!(ax, decompose(μ)...; linewidth = 3, color = :black)
            f
        end
        begin # * Plot velocity distribution (step size distribution)
            ax = Axis(gs[i, 2]; xlabel = "Step size (px/s)", ylabel = "Frequency",
                      xscale = log10,
                      yscale = log10, limits = ((10^(-2.9), 10^(1.9)), nothing))
            i > 1 && hideydecorations!(ax)
            ax.ygridvisible = true
            bins = exp10.(range(-4, 3, length = 200))
            bincs = (bins[1:(end - 1)] .+ bins[2:end]) ./ 2
            # bins = [a .. b for (a, b) in zip(bins[1:(end - 1)], bins[2:end])]
            stepdist = map(velocity[condition = At(CONDS)]) do x
                m = map(x) do y
                    ToolsArray(fit(Histogram, parent(y)[:], bins).weights,
                               (Dim{:velocity}(bincs),))
                    # g = DimensionalData.groupby(y[:, 1], 𝑡 => Bins(bins)) .|> length
                end |> stack
                m = dropdims(sum(m, dims = :uid), dims = :uid)
            end
            stepdist = sum(stepdist)
            stepdist = stepdist[stepdist .!= 0]

            if PERIOD === FIXATION
                s = stepdist[Dim{:velocity}(0.5 .. 100)]
                X = hcat(ones(Float64, dims(s)), log10.(lookup(s, 1)))
                m = collect(X) \ log10.(s)
                lines!(ax, lookup(s, 1),
                       10 .^ (m[1] .+ m[2] * log10.(lookup(s, 1)));
                       color = :red)
                text!(ax, 3, 1e5; text = "α = $(round(m[2], digits = 2))")
            end
            lines!(ax, lookup(stepdist, 1), parent(stepdist))
        end
    end
    linkyaxes!([contents(contents(gs[i, 1])[1])[1] for i in 1:3]...)
    linkyaxes!([contents(gs[i, 2])[1] for i in 1:3]...)
    linkyaxes!([contents(gs[i, 3])[1] for i in 1:3]...)
    linkyaxes!([contents(gs[i, 4])[1] for i in 1:3]...)
    linkxaxes!([contents(gs[i, 5])[1] for i in 1:3]...)
    linkyaxes!([contents(gs[i, 5])[1] for i in 1:3]...)
    f
end

# begin # * Plot distribution of exit times (proxied by macrosaccades).
#     g = 20
#     idx = 30
#     img = reverse(gallery[g], dims = 1)'
#     axs = [Interval((a .- mean(a))...) for a in extrema.(axes(img))]
#     image(axs..., img)

#     x = gaze[g][idx]
#     m = DimensionalData.metadata(x)
#     x = x[𝑡(m["fix_off"] .. m["tar_on"])]
#     lines!(collect.(eachcol(x[:, :, 1]))...)

#     ss = m["saccade_onsets"]
#     ss = ss[ss .∈ [m["fix_off"] .. m["tar_on"]]]
#     _x = x[𝑡(Near(ss))]
#     scatter!(collect.(eachcol(_x[:, :, 1]))...; markersize = 10, color = :green)

#     ss = m["saccade_offsets"]
#     ss = ss[ss .∈ [m["fix_off"] .. m["tar_on"]]]
#     _x = x[𝑡(Near(ss))]
#     scatter!(collect.(eachcol(_x[:, :, 1]))...; markersize = 10, color = :red)

#     current_figure()
# end

if PERIOD == FIXATION
    save("./plots/hexgrid_fix_dynamics.pdf", f)
else
    save("./plots/hexgrid_dynamics.pdf", f)
end

if false # * Velocity distribution matches saccade size?
    x = index.amp
    x = filter(!ismissing, x)
    bins = exp10.(-1.5:0.04:3.5)
    hist(collect(Iterators.flatten(x)); bins, axis = (; xscale = log10, yscale = log10))
    # density(filter(>(0), log10.(collect(Iterators.flatten(x)))))
end
