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
using Printf

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

function plot_fixation_psd(
    df::DataFrame;
    fs=1000,
    eye=:right,
    filter_freq=250,
    welch_window=256,
    welch_overlap=64,
    nfft=256,
)
    lpf = digitalfilter(Lowpass(filter_freq, fs=fs), Butterworth(4))  

    eye in (:right, :left) || throw(ArgumentError("eye must be :right or :left"))
    0 <= welch_overlap < welch_window ||
        throw(ArgumentError("welch_overlap must be nonnegative and less than welch_window"))
    nfft >= welch_window ||
        throw(ArgumentError("nfft must be at least welch_window"))

    # 1. Filter for valid fixations only
    valid_fix = filter(row -> row.status == :valid, df)
    
    if nrow(valid_fix) == 0
        error("No valid fixations found in the data.")
    end
    
    println("Calculating PSD and SEM across $(nrow(valid_fix)) valid fixations...")
    
    psd_x_list = Vector{Float64}[]
    psd_y_list = Vector{Float64}[]
    freqs = Float64[]
    skipped_short = 0
    
    for row in eachrow(valid_fix)
        eye_data = getproperty(row, eye)
        n_samples = size(eye_data.data, 1)
        if n_samples < welch_window
            skipped_short += 1
            continue
        end
        
        x_pos = eye_data.data[:, 1]
        y_pos = eye_data.data[:, 2]

        x_pos_filtered = filtfilt(lpf, x_pos)
        y_pos_filtered = filtfilt(lpf, y_pos)

        x_vel = gradient(x_pos_filtered, 1/fs)
        y_vel = gradient(y_pos_filtered, 1/fs)
        
        x_centered = x_vel .- mean(x_vel)
        y_centered = y_vel .- mean(y_vel)
        
        px = welch_pgram(
            x_centered,
            welch_window,
            welch_overlap;
            nfft=nfft,
            fs=fs,
            window=hanning,
        )
        py = welch_pgram(
            y_centered,
            welch_window,
            welch_overlap;
            nfft=nfft,
            fs=fs,
            window=hanning,
        )
        
        if isempty(freqs)
            freqs = px.freq
        elseif px.freq != freqs
            error("Welch frequency grid changed between fixations.")
        end
        
        push!(psd_x_list, px.power)
        push!(psd_y_list, py.power)
    end
    
    if isempty(psd_x_list)
        error("No valid fixation is at least welch_window samples long.")
    end
    skipped_short > 0 &&
        println("Skipped $skipped_short valid fixations shorter than welch_window=$welch_window.")

    N = length(psd_x_list)

    # 4. Matrix Conversion & Error Calculation
    psd_x_mat = reduce(hcat, psd_x_list)
    psd_y_mat = reduce(hcat, psd_y_list)
    
    avg_psd_x = vec(mean(psd_x_mat, dims=2))
    avg_psd_y = vec(mean(psd_y_mat, dims=2))
    
    # Calculate 95% Confidence Interval (Z ≈ 1.96 for large N)
    z_score = 1.96
    ci95_x = z_score .* vec(std(psd_x_mat, dims=2)) ./ sqrt(N)
    ci95_y = z_score .* vec(std(psd_y_mat, dims=2)) ./ sqrt(N)
    
    # 5. Filter out 0 Hz 
    valid_idx = freqs .> 0
    f_plot = freqs[valid_idx]
    
    px_plot = avg_psd_x[valid_idx]
    py_plot = avg_psd_y[valid_idx]
    ci_x_plot = ci95_x[valid_idx]
    ci_y_plot = ci95_y[valid_idx]
    
    # Log-safe bounds
    px_lower_err = px_plot .- max.(px_plot .- ci_x_plot, 1e-12)
    py_lower_err = py_plot .- max.(py_plot .- ci_y_plot, 1e-12)
    
    px_ribbon = (px_lower_err, ci_x_plot)
    py_ribbon = (py_lower_err, ci_y_plot)

    peak_idx_x = argmax(px_plot)
    peak_freq_x = f_plot[peak_idx_x]
    
    peak_idx_y = argmax(py_plot)
    peak_freq_y = f_plot[peak_idx_y]

    peak_freq = (peak_freq_x + peak_freq_y) / 2

    power_peak = 1.2*maximum(vcat(px_plot, py_plot))
    y_scale = 1e3
    y_tick_formatter = y -> @sprintf("%.2g", y / y_scale)

    # 6. Plotting
    p = plot(f_plot, px_plot, 
        ribbon=px_ribbon,
        fillalpha=0.3,
        label="X-velocity",
        color="#0072B2",
        xlims=(0, 250),
        ylims=(0, power_peak),
        linewidth=4,
        xlabel="Frequency (Hz)", 
        ylabel="Velocity PSD (10³ (px/s)²/Hz)", 
        title="Power Spectrum with 95% CI \n$N fixations, binocular average",
        size=(1000, 650), 
        titlefontsize=24, guidefontsize=18, tickfontsize=20,
        legendfontsize=20, framestyle=:box, legend=:topright, margin=14Plots.mm,
        grid=true, minorticks=true, yformatter=y_tick_formatter)

    Plots.xticks!(0:50:250)
    # Plots.yticks!(0:50:power_peak)


        
    plot!(p, f_plot, py_plot, 
        ribbon=py_ribbon,
        fillalpha=0.3,
        label="Y-velocity", 
        linewidth=4, 
        color="#D55E00")

    # annotate!([(250.0, plot_max_psd,
    #     text("×10^$y_exponent", :black, 20, :right))])

    vline!(p, [peak_freq], color=:black, linestyle=:dash, linewidth=3, label="Peak Frequency: $(round(peak_freq, digits=2)) Hz")
    return p
end

# === Execution Pipeline ===
df_fixations = load_data(THRESHOLD=0.5, BUFFER=0.05, max_speed=10, fs_file=1000)
psd_plot = plot_fixation_psd(df_fixations, fs=1000, eye=:right)
display(psd_plot)

savefig(psd_plot, plotsdir("power_spectra", "fixation_psd_plot.pdf"))
