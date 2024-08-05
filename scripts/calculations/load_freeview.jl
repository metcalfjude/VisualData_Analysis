#! /bin/bash
# -*- mode: julia -*-
#=
exec julia -t auto --startup-file=no --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
using DelimitedFiles
using CairoMakie
using DSP
using LinearAlgebra
using Term
using StatsBase
using TimeseriesTools
using Foresight
@quickactivate "FractionalSaccades"
import FractionalSaccades as FS
set_theme!(foresight(:physics, :serif))

begin
    files = readdir(datadir("raw", "freeview"); join = true)
    Ss = readdlm.(files, ',', Float32; header = true)
    Ss = Ss .|> first
    Ss = map(Ss) do x
        Timeseries(range(0, length = size(x, 1), step = 1 / 1000), 1:4, x)
    end
    S = Ss[1]
end

begin
    x = S[16000:17000, 1]
    plot(x)
end
begin
    spectrumplot(spectrum(S)[Freq = 1 .. 250])
end

if false
    f = TwoPanel()
    axs = [Axis(f[1, 1], xlabel = "Time (s)", ylabel = "S (px)"),
        Axis(f[1, 2], xlabel = "Time (s)", ylabel = "Y (px)")]
    ts = 5000:20000
    lines!(axs[1], collect(S[ts, 1]), collect(S[ts, 2]), color = times(S)[ts])
    lines!(axs[2], collect(S[ts, 3]), collect(S[ts, 4]), color = times(S)[ts])
    f
end

if false # * Plot autocorrelation function of x
    τ_max = 10000
    r = progressmap(Ss) do S
        crosscor(S[:, 1], S[:, 2], 1:τ_max)
    end
    r = stack(r)
    r = Timeseries((1:τ_max) .* step(S), Dim{:subject}(0:(length(Ss) - 1)), r)
end
if false
    f = Figure()
    ax = Axis(f[1, 1]; limits = (nothing, (-0.1, 0.25)))
    μ = dropdims(mean(r, dims = Dim{:subject}), dims = :subject)
    σ = dropdims(std(r, dims = Dim{:subject}), dims = :subject)
    # bandwidth!(ax, decompose(μ)...; linewidth = collect(σ), color = (:cornflowerblue, 0.3))
    lines!(ax, r[:, 4]; color = :cornflowerblue)
    f
end

begin # * Plot msd
    function msd(X,
                 τs = round.(Int,
                             exp10.(range(0, stop = floor(log10(size(X, 1)) - 1),
                                          length = 100))))
        D = Vector{eltype(X)}(undef, length(τs))
        Threads.@threads for (i, τ) in collect(enumerate(τs))
            D[i] = mean(norm(X[1:(end - τ), :] .- X[(τ + 1):end, :]) .^ 2)
        end
        return Timeseries(τs, D)
    end
    τs = minimum(floor.(log10.(size.(Ss, 1)) .- 1))
    τs = unique(round.(Int, exp10.(range(0, stop = τs, length = 100))))
    _D = progressmap(Ss) do S
        msd(S[:, 1:2], τs)
    end
    D = cat(_D...; dims = Dim{:subject}(0:(length(_D) - 1)))
    D = set(D, Ti => τs * step(Ss[1]))
    dtau = 1e-3 .. 1e-1
    m = map(eachslice(D, dims = 2)) do X
        X = X[Ti = dtau]
        x = log10.(times(X))
        y = log10.(parent(X))
        x = x .- mean(x)
        y = y .- mean(y)
        x \ y
    end
    ms = std(m)
    m̄ = mean(m)
    subt = times(D[Ti = dtau])
    md = Timeseries(subt, exp10.(log10.(subt) .* m̄))
end
begin
    f = OnePanel()
    ax = Axis(f[1, 1]; xscale = log10, yscale = log10, xlabel = L"\tau \textrm{ (s)}",
              ylabel = "Mean-squared displacement")
    μ = dropdims(mean(D, dims = Dim{:subject}), dims = :subject)
    σ = dropdims(std(D, dims = Dim{:subject}), dims = :subject)
    band!(ax, times(μ), collect(μ .- σ), collect(μ .+ σ); color = (:cornflowerblue, 0.3))
    lines!(ax, decompose(μ)...; color = :cornflowerblue)

    lines!(ax, decompose(md * 1e13)...; color = :crimson, linestyle = :dash)
    text!(ax, 0.003, 10^(11.3);
          text = L"\alpha = %$(round(m̄; sigdigits=3)) \pm %$(round(ms; sigdigits=1))")
    wsave("msd.pdf", f)
    wsave("msd.png", f; px_per_unit = 10)
    f
end
