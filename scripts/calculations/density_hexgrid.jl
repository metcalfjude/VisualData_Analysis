using DrWatson
using JLD2
using DataFrames
using CairoMakie
using TimeseriesTools
using Distributions
using StatsBase
using GaussianMixtures
using StableDistributions
using Foresight
set_theme!(foresight(:physics))

file = jldopen(datadir("native", "hexgrid", "human.jld2"))
df = file["df"]
condition = 7 # Wide-spaced hex grid
subd = df[df.cond_id .== condition, :]
files = subd.gaze_file

begin # * Fixation period
    xs = map(files) do f
        x = DrWatson.load(f, "gaze")[:, 2, 1] # Left eye, x position
        T = first(times(x)) .. DimensionalData.metadata(x)["fix_off"]
        x[Ti(T)]
    end
    x = vcat(parent.(xs)...)
    filter!(!isnan, x)
end

begin # * Try fitting a distribution
    D = fit_mle(Laplace, x)
    f = Figure()
    ax = Axis(f[1, 1], limits = ((-50, 50), nothing))
    density!(ax, x)
    lines!(ax, range(extrema(x)...; length = 1000), x -> pdf(D, x))
    f
end
begin
    # ! Patch Distributions
    function StatsBase.fit(::Type{<:Logistic}, x::AbstractArray{T}) where {T <: Real}
        m = median(x)
        s = sqrt(3 * var(x) / π^2)
        Logistic(m, s)
    end
    struct HyperbolicSecant{T <: Real} <: ContinuousUnivariateDistribution
        μ::T
        σ::T
        HyperbolicSecant{T}(µ::T, σ::T) where {T <: Real} = new{T}(µ, σ)
    end
    function Distributions.pdf(d::HyperbolicSecant, x::Real)
        sech(0.5 * π * (x - d.μ) / d.σ) / (2 * d.σ)
    end
    function StatsBase.fit(::Type{<:HyperbolicSecant},
                           x::AbstractArray{T}) where {T <: Real}
        m = mean(x)
        s = std(x)
        HyperbolicSecant{T}(m, s)
    end
end
begin # * Try fitting candidate distributions
    dists = [Normal, Cauchy, Stable, Laplace]
    fits = fit.(dists, [x])
    _xs = range(-50, 50; length = 100)
    mx = mean([_xs[1:(end - 1)], _xs[2:end]])
    ft = fit(Histogram, x, _xs).weights
    ft = ft ./ sum(ft)
    gs = [x -> pdf.([D], x) for D in fits]
    err = map(gs) do g
        mean(((g(mx) .- ft) ./ mean([g(mx), ft])) .^ 2)
    end
end
begin
    f = Figure()
    ax = Axis(f[1, 1]; limits = ((-15, 15), nothing),
              title = "Hex grid fixation (condition = $condition)",
              xlabel = "x", ylabel = "Density")
    _xs = range(-50, 50; length = 1000)
    density!(ax, x; color = (:gray, 0.5), strokecolor = :transparent, boundary = (-20, 20))
    map(fits, gs, err) do D, g, e
        name = split(string(typeof(D)), '{')[1]
        lines!(ax, _xs, x -> pdf(D, x);
               label = name * ": NSSE=" * string(round(e, sigdigits = 2)),
               linewidth = 4, alpha = 0.8)
    end
    # lines!(ax, mx, ft; color = :gray, linewidth = 4, alpha = 0.5)
    axislegend(ax)
    tightlimits!(ax)
    f
end

begin
    ys = map(xs) do x
        x[1:minimum(length.(xs))]
    end
    S = spectrum.(ys)
    S = cat(S...; dims = 2)
    S = nansafe(mean, dims = 2)(S)[:, 1]
    spectrumplot(S[Freq(0 .. 150)])
end
