using DrWatson
@quickactivate :FractionalSaccades

using DataFrames
using DSP
using TimeseriesTools
using DimensionalData
using JLD2
using KernelDensity
using Plots
using Statistics
using LinearAlgebra
using FFTW

include(scriptsdir("calculations/load_data.jl"))

default(
    titlefontsize = 16,
    guidefontsize = 18,
    tickfontsize = 20,
    legendfontsize = 14,
    grid = false,
    framestyle = :box,
    dpi = 300,
    margin = 8Plots.mm,
    legend = :topright,
)

function normalized_autocorrelation(signal::AbstractVector, max_lag_samples::Int)
    centered = Float64.(signal) .- mean(signal)
    variance = sum(abs2, centered)
    variance > 0 || throw(ArgumentError("Cannot calculate autocorrelation for a constant signal."))

    n_samples = length(centered)
    autocorrelation = Vector{Float64}(undef, max_lag_samples + 1)
    for lag in 0:max_lag_samples
        n_pairs = n_samples - lag
        autocorrelation[lag + 1] =
            dot(centered[1:n_pairs], centered[(lag + 1):n_samples]) /
            (n_pairs * variance / n_samples)
    end
    return autocorrelation
end

function fourier_interpolate_acf(
    autocorrelation::AbstractVector,
    interpolation_factor::Int,
)
    interpolation_factor >= 1 ||
        throw(ArgumentError("interpolation_factor must be at least 1"))
    interpolation_factor == 1 && return Float64.(autocorrelation)

    # Mirror the non-negative lags so Fourier interpolation preserves the
    # autocorrelation's even symmetry.
    symmetric_acf = vcat(reverse(autocorrelation), autocorrelation)
    n_original = length(symmetric_acf)
    n_interpolated = interpolation_factor * n_original
    spectrum = fftshift(fft(symmetric_acf))
    padding = n_interpolated - n_original
    padded_spectrum = vcat(
        zeros(ComplexF64, padding ÷ 2),
        spectrum,
        zeros(ComplexF64, padding - padding ÷ 2),
    )
    interpolated = real.(ifft(ifftshift(padded_spectrum))) .* interpolation_factor

    zero_lag_index = length(autocorrelation) * interpolation_factor + 1
    interpolated_lag_count = (length(autocorrelation) - 1) * interpolation_factor
    return interpolated[zero_lag_index:(zero_lag_index + interpolated_lag_count)]
end

function acf_minimum_and_crossover(lags, acf)
    minimum_index = argmin(acf[2:end]) + 1
    minimum_lag = lags[minimum_index]
    minimum_value = acf[minimum_index]

    crossover_index = findfirst(
        i -> acf[i] < 0 <= acf[i + 1],
        minimum_index:(length(acf) - 1),
    )
    crossover_index === nothing &&
        return (minimum_lag, minimum_value, nothing)

    # Linear interpolation gives a sub-sample estimate of the zero crossing.
    i = crossover_index
    crossover_lag = lags[i] +
        (lags[i + 1] - lags[i]) * (-acf[i]) / (acf[i + 1] - acf[i])
    return (minimum_lag, minimum_value, crossover_lag)
end

function plot_fixation_acf(
    df::DataFrame;
    fs=1000,
    eye=:right,
    filter_freq=250,
    max_lag=0.5,
    interpolation_factor=1,
)
    eye in (:right, :left) || throw(ArgumentError("eye must be :right or :left"))
    max_lag > 0 || throw(ArgumentError("max_lag must be positive"))
    interpolation_factor >= 1 ||
        throw(ArgumentError("interpolation_factor must be at least 1"))

    max_lag_samples = round(Int, max_lag * fs)
    max_lag_samples > 0 || throw(ArgumentError("max_lag must contain at least one sample"))

    lpf = digitalfilter(Lowpass(filter_freq, fs=fs), Butterworth(4))
    valid_fix = filter(row -> row.status == :valid, df)

    nrow(valid_fix) > 0 || error("No valid fixations found in the data.")
    println("Calculating fixation autocorrelation and SEM across $(nrow(valid_fix)) valid fixations...")

    acf_x_list = Vector{Float64}[]
    acf_y_list = Vector{Float64}[]
    skipped_short = 0
    skipped_constant = 0

    for row in eachrow(valid_fix)
        eye_data = getproperty(row, eye)
        n_samples = size(eye_data.data, 1)
        if n_samples <= max_lag_samples
            skipped_short += 1
            continue
        end

        x_pos = eye_data.data[:, 1]
        y_pos = eye_data.data[:, 2]
        x_pos_filtered = filtfilt(lpf, x_pos)
        y_pos_filtered = filtfilt(lpf, y_pos)
        x_velocity = gradient(x_pos_filtered, 1 / fs)
        y_velocity = gradient(y_pos_filtered, 1 / fs)

        try
            push!(acf_x_list, normalized_autocorrelation(x_velocity, max_lag_samples))
            push!(acf_y_list, normalized_autocorrelation(y_velocity, max_lag_samples))
        catch error
            if error isa ArgumentError
                skipped_constant += 1
            else
                rethrow()
            end
        end
    end

    isempty(acf_x_list) && error("No valid fixation is longer than max_lag=$(max_lag) seconds.")
    skipped_short > 0 &&
        println("Skipped $skipped_short valid fixations shorter than max_lag=$(max_lag) seconds.")
    skipped_constant > 0 &&
        println("Skipped $skipped_constant fixations with constant velocity.")

    acf_x_mat = reduce(hcat, acf_x_list)
    acf_y_mat = reduce(hcat, acf_y_list)
    n_fixations = size(acf_x_mat, 2)

    mean_acf_x = fourier_interpolate_acf(
        vec(mean(acf_x_mat, dims=2)),
        interpolation_factor,
    )
    mean_acf_y = fourier_interpolate_acf(
        vec(mean(acf_y_mat, dims=2)),
        interpolation_factor,
    )
    ci95_x = fourier_interpolate_acf(
        1.96 .* vec(std(acf_x_mat, dims=2)) ./ sqrt(n_fixations),
        interpolation_factor,
    )
    ci95_y = fourier_interpolate_acf(
        1.96 .* vec(std(acf_y_mat, dims=2)) ./ sqrt(n_fixations),
        interpolation_factor,
    )
    lags = collect(0:(max_lag_samples * interpolation_factor)) /
        (fs * interpolation_factor)
    lags = lags .* 1000  # Convert to milliseconds

    minimum_x = acf_minimum_and_crossover(lags, mean_acf_x)
    minimum_y = acf_minimum_and_crossover(lags, mean_acf_y)
    println(
        "X-velocity ACF minimum: $(round(minimum_x[2], digits=4)) at " *
        "$(round(minimum_x[1], digits=3)) ms; crossover: " *
        (minimum_x[3] === nothing ? "not found" : "$(round(minimum_x[3], digits=3)) ms"),
    )
    println(
        "Y-velocity ACF minimum: $(round(minimum_y[2], digits=4)) at " *
        "$(round(minimum_y[1], digits=3)) ms; crossover: " *
        (minimum_y[3] === nothing ? "not found" : "$(round(minimum_y[3], digits=3)) ms"),
    )

    p = plot(
        lags,
        mean_acf_x,
        ribbon=(ci95_x, ci95_x),
        fillalpha=0.3,
        label="X-velocity",
        color="#0072B2",
        linewidth=4,
        xlabel="Lag (ms)",
        ylabel="Normalized autocorrelation",
        title="Fixation Autocorrelation with 95% CI\n$n_fixations fixations, $(eye) eye",
        size=(1000, 650),
        titlefontsize=24,
        guidefontsize=18,
        tickfontsize=20,
        legendfontsize=20,
        framestyle=:box,
        legend=:topright,
        margin=14Plots.mm,
        grid=true,
        minorticks=true,
    )

    plot!(
        p,
        lags,
        mean_acf_y,
        ribbon=(ci95_y, ci95_y),
        fillalpha=0.3,
        label="Y-velocity",
        color="#D55E00",
        linewidth=4,
    )
    hline!(p, [0], color=:black, linestyle=:dash, linewidth=2, label=nothing)

    scatter!(
        p,
        [minimum_x[1]],
        [minimum_x[2]],
        color="#0072B2",
        markersize=7,
        label="X minimum",
    )
    scatter!(
        p,
        [minimum_y[1]],
        [minimum_y[2]],
        color="#D55E00",
        markersize=7,
        label="Y minimum",
    )
    if minimum_x[3] !== nothing
        scatter!(
            p,
            [minimum_x[3]],
            [0],
            color="#0072B2",
            marker=:xcross,
            markersize=8,
            label="X crossover",
        )
    end
    if minimum_y[3] !== nothing
        scatter!(
            p,
            [minimum_y[3]],
            [0],
            color="#D55E00",
            marker=:xcross,
            markersize=8,
            label="Y crossover",
        )
    end
    return p
end

# === Execution Pipeline ===
df_fixations = load_data(THRESHOLD=0.5, BUFFER=0.05, max_speed=10, fs_file=1000)
acf_plot = plot_fixation_acf(
    df_fixations,
    fs=1000,
    eye=:right,
    max_lag=0.05,
    interpolation_factor=4,
)
display(acf_plot)

mkpath(plotsdir("autocorrelation"))
savefig(acf_plot, plotsdir("autocorrelation", "fixation_acf_plot.pdf"))
