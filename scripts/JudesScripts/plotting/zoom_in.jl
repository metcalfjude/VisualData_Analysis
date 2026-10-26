using DrWatson
@quickactivate :FractionalSaccades

include(scriptsdir("JudesScripts/total_search/load_search_data.jl"))

using CairoMakie
using DataFrames
using DimensionalData
using TimeseriesTools
using DSP
using JLD2

# ==========================================
# 1. LOAD AND PREP DATA
# ==========================================
df = load_search_data(; max_nan_pct=0.20, pin_first_saccade=false)

# Filter for valid fixations and isolate a subject
valid_df = filter(:status => ==(:valid), df)
target_subject = unique(valid_df.subject_id)[1] 
subj_df = filter(:subject_id => ==(target_subject), valid_df)

# Extract the right eye timeseries from the very first valid fixation
fixation = first(subj_df)
r_eye = fixation.right

# Recover the annotated saccade onsets for this trial. The loader's
# `start_idx` identifies the beginning of the plotted trial window.
raw_index, raw_trial_times, raw_saccade_onsets = jldopen(
    datadir("native", "hexgrid", "gaze.jld2"), "r") do file
    raw_index = file["index"]
    raw_trial = first(filter(:uid => ==(fixation.uid), raw_index))
    trial_times = collect(times(file[raw_trial.gaze_file]))
    (raw_index, trial_times, raw_trial.sac_onsets)
end

# Extract underlying time vector and gaze coordinates
t_vals = collect(times(r_eye))
x_vals = r_eye[:, 1].data
y_vals = r_eye[:, 2].data

# Create a mask of indices where both gaze coordinates are valid
valid_mask = .!(isnan.(x_vals) .| isnan.(y_vals))
valid_t_vals = t_vals[valid_mask]
valid_x_vals = x_vals[valid_mask]
valid_y_vals = y_vals[valid_mask]

# ==========================================
# 2. DEFINE TARGET ZOOM WINDOW
# ==========================================
# Specify the exact times you want to zoom into across ALL plots
target_start_time = 6.4
target_end_time = 6.75  

start_idx = clamp(searchsortedfirst(valid_t_vals, target_start_time), 1, length(valid_t_vals))
end_idx = clamp(searchsortedlast(valid_t_vals, target_end_time), start_idx, length(valid_t_vals))

start_xlim = [valid_t_vals[1], valid_t_vals[end]]
end_xlim = [valid_t_vals[start_idx], valid_t_vals[end_idx]]

# ==========================================
# 3. SIGNAL PROCESSING (ALL FILTERS)
# ==========================================
fs = 1000

plotted_t_vals = valid_t_vals[start_idx:end_idx]
plotted_x_vals = valid_x_vals[start_idx:end_idx]
plotted_y_vals = valid_y_vals[start_idx:end_idx]

# A. Smoothing filter (200Hz) for the main animation
lp_200 = digitalfilter(Lowpass(200, fs=fs), Butterworth(4))
filtered_x_vals = filtfilt(lp_200, valid_x_vals)

# B. Decomposition filters for the PDF
cutoff = 20
lp = digitalfilter(Lowpass(cutoff, fs=fs), Butterworth(4))
bp = digitalfilter(Bandpass(cutoff, 120, fs=fs), Butterworth(4))

c, m = [ones(length(plotted_x_vals)) plotted_t_vals] \ plotted_x_vals # Linear drift

de_drifted_x_vals = @. (plotted_x_vals - c) - m * plotted_t_vals

low_band  = filtfilt(lp, de_drifted_x_vals) .+ c .+ m .* plotted_t_vals
high_band = filtfilt(bp, de_drifted_x_vals)
residual = plotted_x_vals .- low_band .- high_band

# ==========================================
# 4A. TOTAL GAZE TRAJECTORY WITH STEP PEAKS
# ==========================================
saccade_onset_times = raw_trial_times[
    filter(i -> fixation.start_idx <= i <= fixation.end_idx, raw_saccade_onsets)]

# Add any missed saccades here using their onset times in seconds.
manual_saccade_onset_times = Float64[4.09,4.74,6.75,8]

saccade_onset_times = sort(unique(filter(
    t -> first(valid_t_vals) <= t <= last(valid_t_vals),
    vcat(saccade_onset_times, manual_saccade_onset_times))))

fig_trajectory = Figure(size = (800, 400))
ax_trajectory_x = Axis(fig_trajectory[1, 1];
    title = "Right eye X gaze trajectory with saccade onsets",
    xlabel = "Time (s)",
    ylabel = "X position (px)"
    )

lines!(ax_trajectory_x, valid_t_vals, valid_x_vals;
       linewidth = 2.0, color=:indigo)
save(plotsdir("plotting", "total_gaze_trajectory_no_peaks.pdf"), fig_trajectory)

fig_trajectory = Figure(size = (800, 400))
ax_trajectory_x = Axis(fig_trajectory[1, 1];
    title = "X gaze trajectory with saccade onsets",
    xlabel = "Time (s)",
    ylabel = "X position (px)"
    )

lines!(ax_trajectory_x, valid_t_vals, valid_x_vals;
       linewidth = 2.0, color=:indigo)

for onset_time in saccade_onset_times
    CairoMakie.vlines!(ax_trajectory_x, onset_time;
                       color = (:black, 0.75), linestyle = :dash, linewidth = 1.5)
end
save(plotsdir("plotting", "total_gaze_trajectory_peaks.pdf"), fig_trajectory)

# ==========================================
# 5. PART A: ANIMATED ZOOM (SINGLE PLOT)
# ==========================================
fig_anim = Figure(size = (800, 400))
ax_anim = Axis(fig_anim[1, 1], 
    title = "Right Eye X-Position (Fixation Zoom)",
    xlabel = "Time (s)", 
    ylabel = "Gaze Position (px/deg)",
    xgridvisible = false, # Turns off vertical gridlines
    ygridvisible = false  # Turns off horizontal gridlines
)

lines!(ax_anim, valid_t_vals, valid_x_vals, linewidth=1, color=(:indigo, 0.3), label="Raw")
lines!(ax_anim, valid_t_vals, filtered_x_vals, linewidth=2, color=:indigo, label="Filtered")
axislegend(ax_anim)

x_size = maximum(valid_x_vals) - minimum(valid_x_vals)

start_ylim = [minimum(valid_x_vals) - 0.1*x_size, maximum(valid_x_vals) + 0.1*x_size]
valid_zoom_y = valid_x_vals[start_idx:end_idx]

if isempty(valid_zoom_y)
    end_ylim = start_ylim 
else
    y_size = maximum(valid_zoom_y) - minimum(valid_zoom_y)
    end_ylim = [minimum(valid_zoom_y) - 0.1*y_size, maximum(valid_zoom_y) + 0.1*y_size]
end

frames = 1:480

record(fig_anim, plotsdir("plotting", "fixation_zoom.mp4"), frames; framerate = 60) do i
    t = (i - 1) / (length(frames) - 1)
    
    ts = 0.25
    
    if t < ts
        # Fast acceleration phase
        ease_t = (1 / ts) * t^2
    else
        # Long, slow deceleration phase
        ease_t = 1 - (1 / (1 - ts)) * (1 - t)^2
    end
    
    cur_x = start_xlim .+ ease_t .* (end_xlim .- start_xlim)
    cur_y = start_ylim .+ ease_t .* (end_ylim .- start_ylim)
    
    CairoMakie.xlims!(ax_anim, cur_x[1], cur_x[2])
    CairoMakie.ylims!(ax_anim, cur_y[1], cur_y[2])
end

# ==========================================
# 6. PART B: STATIC DECOMPOSITION (PDF)
# ==========================================
fig_pdf = Figure(size = (800, 800))

ax_raw = Axis(fig_pdf[1, 1], title = "Raw Gaze Position", ylabel = "Amplitude (px)")
ax_low = Axis(fig_pdf[2, 1], title = "Low Frequency (<$cutoff Hz) and 200 Hz Low-Pass", ylabel = "Amplitude")
ax_high = Axis(fig_pdf[3, 1], title = "High Frequency ($cutoff-120 Hz)", ylabel = "Amplitude")
ax_res = Axis(fig_pdf[4, 1], title = "Residual", xlabel = "Time (s)", ylabel = "Amplitude")

CairoMakie.scatter!(ax_raw, plotted_t_vals, plotted_x_vals, markersize=4, color=:black)
CairoMakie.scatter!(ax_low, plotted_t_vals, low_band, markersize=4, color=:indigo, label="<$cutoff Hz")
CairoMakie.scatter!(
    ax_low,
    plotted_t_vals,
    filtered_x_vals[start_idx:end_idx],
    markersize=3,
    color=:darkorange,
    label="200 Hz low-pass",
)
CairoMakie.scatter!(ax_high, plotted_t_vals, high_band, markersize=4, color=:firebrick)
CairoMakie.scatter!(ax_res, plotted_t_vals, residual, markersize=4, color=:gray50)
CairoMakie.axislegend(ax_low, position=:rt)
CairoMakie.linkxaxes!(ax_raw, ax_low, ax_high, ax_res)

function get_padded_ylims(data_array)
    if isempty(data_array)
        return start_ylim # Fallback if window is somehow empty
    end
    d_min, d_max = minimum(data_array), maximum(data_array)
    pad = max((d_max - d_min) * 0.1, 0.1) 
    return [d_min - pad, d_max + pad]
end

axes_list = [ax_raw, ax_low, ax_high, ax_res]
data_list = [plotted_x_vals, low_band, high_band, residual]

for (ax, data) in zip(axes_list, data_list)
    CairoMakie.xlims!(ax, end_xlim[1], end_xlim[2])
    
    windowed_data = data
    y_lims = get_padded_ylims(windowed_data)
    CairoMakie.ylims!(ax, y_lims[1], y_lims[2])
end

CairoMakie.ylims!(ax_high, -2, 2) # Set fixed y-limits for the high-frequency band and residual to compare amplitudes
CairoMakie.ylims!(ax_res, -2, 2) 
save(plotsdir("plotting", "fixation_decomposition_zoomed.pdf"), fig_pdf)

# ==========================================
# 7. plotting fixation iteratively
# ==========================================
data_list = [plotted_x_vals, 
    filtered_x_vals[start_idx:end_idx],
    gradient(filtered_x_vals, 1/fs)[start_idx:end_idx]]

zoom_t_vals = plotted_t_vals

plot_titles = ["Raw X Position", "Filtered X Gaze Position (<200Hz)", "Gaze X Velocity"]

for i in eachindex(data_list)
    ylabel = i == 3 ? "Velocity (px/s)" : "Position (px)"
    fig = Figure(size = (800, 400))
    ax = Axis(fig[1, 1];
        title = plot_titles[i],
        xlabel = "Time (s)",
        ylabel = ylabel,
    )
    lines!(ax, zoom_t_vals, data_list[i]; linewidth=2.0, color=:indigo)
    save(plotsdir("plotting", "gaze_analysis_plot_$(i).pdf"), fig)
end

# ==========================================
# 8. VELOCITY WITH MANUAL SINUSOIDAL ENVELOPE
# ==========================================
zoom_velocity = gradient(filtered_x_vals, 1/fs)[start_idx:end_idx]

# --- Manual Envelope Parameters ---
amp = 500.0          # Amplitude (px/s) - adjust to match your velocity peaks
freq = 1.8          # Frequency (Hz) - keep < 4Hz
freq_effective = freq * 2  # Convert to radians for sine function
t_zero = 6.65       # Time (s) where the envelope intersects zero
# ----------------------------------

# Generate the abstract sine wave. 
# Because sin(0) = 0, the wave will intersect zero exactly at t_zero.
abstract_sine = amp .* sin.(2 * π * freq .* (zoom_t_vals .- t_zero))

# Take the absolute value to form the symmetric upper and lower bounds
# (If you prefer the braided look of a raw sine wave, just remove abs() )
manual_envelope = abs.(abstract_sine)

fig_env = Figure(size = (800, 400),
    titlefontsize = 24,
    guidefontsize = 18,          
    tickfontsize = 20,
    legendfontsize = 16,
    grid = false)

ax_env = Axis(fig_env[1, 1];
    title = "Gaze X Velocity with Envelope",
    xlabel = "Time (s)",
    ylabel = "Velocity (px/s)"
)

# Plot the base velocity exactly like plot 3
lines!(ax_env, zoom_t_vals, zoom_velocity; linewidth=2.0, color=:indigo, label="Fixation Velocity")

# Overlay the manual envelope (positive and negative bounds)
lines!(ax_env, zoom_t_vals, manual_envelope; linewidth=3.0, linestyle=:dash, color=:firebrick, label="Sinusoidal Envelope (f = $freq_effective Hz)")
lines!(ax_env, zoom_t_vals, .-manual_envelope; linewidth=3.0, linestyle=:dash, color=:firebrick)

axislegend(ax_env, position=:rt)

save(plotsdir("plotting", "gaze_analysis_plot_4_manual_envelope.pdf"), fig_env)
