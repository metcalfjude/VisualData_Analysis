using DrWatson
@quickactivate :FractionalSaccades

using DataFrames
using DSP
using TimeseriesTools
using DimensionalData
using JLD2
using KernelDensity
using Plots
using Statistics # Required for mean() and std()

include("../load_data.jl")

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

function plot_fixation_psd(df::DataFrame; fs=1000, eye=:right, filter_freq = 150)
    lpf = digitalfilter(Lowpass(filter_freq, fs=fs), Butterworth(4))  

    # 1. Filter for valid fixations only
    valid_fix = filter(row -> row.status == :valid, df)
    
    if nrow(valid_fix) == 0
        error("No valid fixations found in the data.")
    end
    
    N = nrow(valid_fix)
    println("Calculating PSD and SEM across $N valid fixations...")
    
    psd_x_list = Vector{Float64}[]
    psd_y_list = Vector{Float64}[]
    freqs = Float64[]
    
    for row in eachrow(valid_fix)
        eye_right = row.right 
        eye_left = row.left
        
        x_pos = eye_right.data[:, 1]# + eye_left.data[:, 1]  # Combine both eyes' x positions
        y_pos = eye_right.data[:, 2]# + eye_left.data[:, 2]  # Combine both eyes' y positions

        x_pos_filtered = filtfilt(lpf, x_pos)[200:end-100]
        y_pos_filtered = filtfilt(lpf, y_pos)[200:end-100]

        x_vel = gradient(x_pos_filtered, 1/fs)
        y_vel = gradient(y_pos_filtered, 1/fs)
        
        x_centered = x_vel .- mean(x_vel)
        y_centered = y_vel .- mean(y_vel)
        
        px = periodogram(x_centered, fs=fs, window=hanning)
        py = periodogram(y_centered, fs=fs, window=hanning)
        
        if isempty(freqs)
            freqs = px.freq
        end
        
        push!(psd_x_list, px.power)
        push!(psd_y_list, py.power)
    end
    
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
    
    # 6. Plotting
    p = plot(f_plot, px_plot, 
        ribbon=px_ribbon,
        fillalpha=0.3,
        label="X-Velocity",
        # xscale=:log10, 
        # yscale=:log10, 
        xlims=(0, 200),
        linewidth=2,
        color=:blue,
        xlabel="Frequency (Hz)", 
        ylabel="Velocity PSD ((px/s)²/Hz)", 
        title="Power Spectrum with 95% CI \n$N fixations, binocular average",
        grid=true,
        minorticks=true,
        legend=:topright)
        
    plot!(p, f_plot, py_plot, 
        ribbon=py_ribbon,
        fillalpha=0.3,
        label="Y-Velocity", 
        linewidth=2, 
        color=:darkorange)

    vline!(p, [peak_freq], color=:black, linestyle=:dash, linewidth=3, label="Peak: $(round(peak_freq, digits=2)) Hz")
        
    return p
end

# === Execution Pipeline ===
df_fixations = load_data(THRESHOLD=1.0, BUFFER=0.01, max_speed=3., fs_file=1000)
psd_plot = plot_fixation_psd(df_fixations, fs=1000, eye=:right)
display(psd_plot)

savefig(psd_plot, plotsdir("power_spectra", "fixation_psd_plot.pdf"))
