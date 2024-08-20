using StableDistributions
using CairoMakie
using Autocorrelations
using TimeseriesTools

α = 1.2
d = Stable(α)
L = rand(d, 10000, 1000) .* (dt)^(1 / α)
dt = 0.01
ts = range(dt, length = size(L, 1), step = dt)
x = cumsum(L; dims = 1)
x = Timeseries(ts, 1:size(L, 2), x)
y = set(x, cumsum(randn(size(x)) .* sqrt(dt); dims = 1))

begin
    ms = map(eachcol(x)) do z
        m = msdist(z)[2:1000]
        last(hcat(ones(length(m)), log10.(times(m))) \ log10.(m |> parent))
    end
end

begin
    using StatsBase
    mu = dropdims(mean(x, dims = 2); dims = 2)
    sigma = dropdims(mapslices(iqr, x, dims = 2); dims = 2).^2

    muy = dropdims(median(y, dims = 2); dims = 2)
    sigmay = dropdims(mapslices(iqr, y, dims = 2); dims = 2).^2
end
begin
    f = Figure()
    ax = Axis(f[1, 1])
    band!(ax, times(mu), parent(mu .- sigma), mu .+ sigma, color = (:blue, 0.2))
    lines!(ax, decompose(mu)...)

    band!(ax, times(muy), parent(muy .- sigmay), muy .+ sigmay, color = (:red, 0.2))
    lines!(ax, decompose(muy)...)
    f
end
begin
    mm = first([log10.(times(sigma)) ones(length(sigma))] \ log10.(sigma |> parent))
    mmy = first([log10.(times(sigmay)) ones(length(sigmay))] \ log10.(sigmay |> parent))
end

begin
    f = Figure()
    ax = Axis(f[1, 1]; xscale = log10, yscale = log10)
    lines!(ax, decompose(sigma)...)
    lines!(ax, decompose(sigmay)...)
    lines!(ax, times(sigma), exp10.(log10.(times(sigma)).*mm))
    f
end
