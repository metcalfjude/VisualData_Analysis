using DrWatson
@quickactivate :FractionalSaccades

using KernelDensity
using DSP
using Plots
using FFTW
using Statistics
using Random
using HDF5
using CSV
using DataFrames
using Polynomials
FractionalSaccades.@preamble

file = jldopen(datadir("native", "hexgrid", "gaze.jld2"), "r")
index = file["index"]
HEXCONDS = 1:12

THRESHOLD = 1.00                # window
BUFFER = 0.001                  # safety buffer to avoid capturing the final acceleration

fs::Int = 1000                  # Sampling frequency
nfft::Int = 1024                # nfft >= n_samples = 1000
nw::Float64 = 2.5               # time-halfbandwidth product

ntapers = floor(Int, 2 * nw) - 1 

frequencies = rfftfreq(nfft, fs)
n_freq = length(frequencies)

begin 
    xs = [only(unique(r.cond_id)) => ToolsArray(getindex.([file], r.gaze_file),
    (Dim{:uid}(r.uid),))
    for r in DataFrames.groupby(index, :cond_id)]
    xs = ToolsArray(last.(xs), (Dim{:condition}(first.(xs)),))
    gaze = filter.(x -> !all(isnan.(x)), xs)
end

# Calculate the exact number of samples required based on THRESHOLD and fs
req_samples = floor(Int, THRESHOLD * fs)

results = map(gaze) do condition_trials
    processed_trials = map(condition_trials) do trial
        meta = DimensionalData.metadata(trial)
        onsets = meta["saccade_onsets"]
        offsets = meta["saccade_offsets"]
        
        # Use a typed NamedTuple array for much better Julia performance
        fixations = NamedTuple[]
        
        for i in 1:(length(onsets) - 1)
            fix_start = offsets[i]
            fix_end = onsets[i+1]
            dur = fix_end - fix_start
            
            if dur >= THRESHOLD + BUFFER
                time_range = 𝑡(fix_end - THRESHOLD - BUFFER .. fix_end - BUFFER)
                
                try
                    r_eye = trial[time_range, Dim{:Eye}(At(:right))]
                    l_eye = trial[time_range, Dim{:Eye}(At(:left))]
                    
                    # Use req_samples instead of a hardcoded 200
                    if !any(isnan, r_eye) && !any(isnan, l_eye) && size(r_eye, 1) == req_samples
                        push!(fixations, (right=r_eye, left=l_eye, status=:valid))
                    else
                        push!(fixations, (right=nothing, left=nothing, status=:rejected_nan_or_short))
                    end
                catch
                    push!(fixations, (right=nothing, left=nothing, status=:rejected_bounds))
                end
            else
                push!(fixations, (right=nothing, left=nothing, status=:rejected_duration))
            end
        end
        return fixations
    end
    return collect(Iterators.flatten(processed_trials))
end

# Flatten one more time to get all fixations from all conditions
all_trials = collect(Iterators.flatten(results))
total_count = length(all_trials)

# Filter valid trials
valid_trials = filter(t -> t.status === :valid, all_trials)
n_trials = length(valid_trials)

println("Successfully extracted $n_trials valid fixation windows.")

rejected_count = total_count - n_trials
println("Rejected $(round(rejected_count/total_count * 100, digits=2))% of trials.")


# Preallocate matching DSP.jl's native shape (channels, channels, frequencies, trials)
all_spectra_x = zeros(ComplexF64, 2, 2, n_freq, n_trials)
all_spectra_y = zeros(ComplexF64, 2, 2, n_freq, n_trials)

n_time = Int(THRESHOLD * fs)

# Preallocate
x_trials = zeros(Float64, n_time, 2, n_trials)
y_trials = zeros(Float64, n_time, 2, n_trials)

fc = 5              # cutoff frequency of 10Hz
order = 4           # Butterworth order

hp_filter = digitalfilter(Highpass(fc, fs=fs), Butterworth(order))

Threads.@threads for i in 1:n_trials
    trial = valid_trials[i]

    # Access the 'left' and 'right' properties of NamedTuple, 
    # then grab column 1 (X) and column 2 (Y)
    left_x  = Float64.(trial.left[:, 1])
    right_x = Float64.(trial.right[:, 1])
    left_y  = Float64.(trial.left[:, 2])
    right_y = Float64.(trial.right[:, 2])

    # Offsetting and weak high-pass filtering
    # Also implemented polynomial fitting, not as helpful
    left_x  = remove_offset(left_x)
    right_x = remove_offset(right_x)
    left_y  = remove_offset(left_y)
    right_y = remove_offset(right_y)

    # # High-Pass filtering directly
    # left_x  = filtfilt(hp_filter, left_x)
    # right_x = filtfilt(hp_filter, right_x)
    # left_y  = filtfilt(hp_filter, left_y)
    # right_y = filtfilt(hp_filter, right_y)

    ## redefine as the steps
    left_x  = gradient(left_x, 1/fs)
    right_x = gradient(right_x, 1/fs)
    left_y  = gradient(left_y, 1/fs)
    right_y = gradient(right_y, 1/fs)

    x_trials[:,1,i] = left_x
    x_trials[:,2,i] = right_x
    y_trials[:,1,i] = left_y
    y_trials[:,2,i] = right_y

    # matrix formatting for DSP.jl
    x_mat_raw = Matrix{Float64}(hcat(left_x, right_x)')
    y_mat_raw = Matrix{Float64}(hcat(left_y, right_y)')

    cs_x = mt_cross_power_spectra(x_mat_raw, fs=fs, nw=nw, ntapers=ntapers, nfft=nfft)
    cs_y = mt_cross_power_spectra(y_mat_raw, fs=fs, nw=nw, ntapers=ntapers, nfft=nfft)

    all_spectra_x[:, :, :, i] = cs_x.power
    all_spectra_y[:, :, :, i] = cs_y.power
end

# Find trials that successfully computed without producing any NaNs
good_fft_mask = [!any(isnan, all_spectra_x[:, :, :, i]) && !any(isnan, all_spectra_y[:, :, :, i]) for i in 1:n_trials]

# Filter down to only perfectly clean spectra
clean_spectra_x = all_spectra_x[:, :, :, good_fft_mask]
clean_spectra_y = all_spectra_y[:, :, :, good_fft_mask]
n_clean_trials = size(clean_spectra_x, 4)

println("Dropped $(n_trials - n_clean_trials) trials due to FFT NaNs.")

# Redefine frequencies and n_freq to skip the 0 Hz bin
frequencies = frequencies[2:end]
n_freq = length(frequencies)

# Slicing out the components (starting at index 2 to skip 0 Hz)
### X arrays
Pll_x = real.(@view clean_spectra_x[1, 1, 2:end, :]) 
Prr_x = real.(@view clean_spectra_x[2, 2, 2:end, :]) 
Plr_x = @view clean_spectra_x[1, 2, 2:end, :]        

### Y arrays
Pll_y = real.(@view clean_spectra_y[1, 1, 2:end, :]) 
Prr_y = real.(@view clean_spectra_y[2, 2, 2:end, :]) 
Plr_y = @view clean_spectra_y[1, 2, 2:end, :]

mean_Pll_x = mean(Pll_x, dims=2)[:, 1]         # Note that the slicing here converts the n_freqs×1 Matrix to a 1D Vector
mean_Prr_x = mean(Prr_x, dims=2)[:, 1]
mean_Plr_x = mean(Plr_x, dims=2)[:, 1]

mean_Pll_y = mean(Pll_y, dims=2)[:, 1]
mean_Prr_y = mean(Prr_y, dims=2)[:, 1]
mean_Plr_y = mean(Plr_y, dims=2)[:, 1]

coherence_x = abs2.(mean_Plr_x) ./ (mean_Pll_x .* mean_Prr_x)
coherence_y = abs2.(mean_Plr_y) ./ (mean_Pll_y .* mean_Prr_y)

n_boot = 10_000 # Number of bootstrap iterations
x_boot_coherence = zeros(n_freq, n_boot)
y_boot_coherence = zeros(n_freq, n_boot)

Threads.@threads for b in 1:n_boot
    # Draw from the clean pool!
    x_idx = rand(1:n_clean_trials, n_clean_trials)
    y_idx = rand(1:n_clean_trials, n_clean_trials)

    # Using @views should reduce memory allocation
    x_boot_mean_Pll = mean(@view(Pll_x[:, x_idx]), dims=2)[:, 1]
    x_boot_mean_Prr = mean(@view(Prr_x[:, x_idx]), dims=2)[:, 1]
    x_boot_mean_Plr = mean(@view(Plr_x[:, x_idx]), dims=2)[:, 1]

    y_boot_mean_Pll = mean(@view(Pll_y[:, y_idx]), dims=2)[:, 1]
    y_boot_mean_Prr = mean(@view(Prr_y[:, y_idx]), dims=2)[:, 1]
    y_boot_mean_Plr = mean(@view(Plr_y[:, y_idx]), dims=2)[:, 1]

    # Calculate coherence
    x_boot_coherence[:, b] = abs2.(x_boot_mean_Plr) ./ (x_boot_mean_Pll .* x_boot_mean_Prr)
    y_boot_coherence[:, b] = abs2.(y_boot_mean_Plr) ./ (y_boot_mean_Pll .* y_boot_mean_Prr)
end

# Bootstrap Standard Error
# (standard deviation of the bootstrap distribution)
x_sem_coherence = std(x_boot_coherence, dims=2)[:, 1]
y_sem_coherence = std(y_boot_coherence, dims=2)[:, 1]

# Helper function to filter out stray NaNs before calculating the quantile
function safe_quantile(v, p)
    clean_v = filter(!isnan, v)
    return isempty(clean_v) ? NaN : quantile(clean_v, p)
end

x_lower_ci = [safe_quantile(x_boot_coherence[f, :], 0.025) for f in 1:n_freq]
x_upper_ci = [safe_quantile(x_boot_coherence[f, :], 0.975) for f in 1:n_freq]

y_lower_ci = [safe_quantile(y_boot_coherence[f, :], 0.025) for f in 1:n_freq]
y_upper_ci = [safe_quantile(y_boot_coherence[f, :], 0.975) for f in 1:n_freq]

# Plotting ribbons
err_lower_x = coherence_x .- x_sem_coherence
err_upper_x = x_sem_coherence .- coherence_x

err_lower_y = coherence_y .- y_sem_coherence
err_upper_y = y_sem_coherence .- coherence_y

p_x = Plots.plot(frequencies, coherence_x,
           ribbon=(err_lower_x, err_upper_x), 
           fillalpha=0.3,
           xlabel="Frequency (Hz)", ylabel="Velocity Coherence",
           title="Horizontal (X) Coherence", 
           label="Mean Coherence", color=:blue, lw=2)

p_y = Plots.plot(frequencies, coherence_y, 
           ribbon=(err_lower_y, err_upper_y), fillalpha=0.3,
           xlabel="Frequency (Hz)", ylabel="Velocity Coherence",
           title="Vertical (Y) Coherence", 
           label="Mean Coherence", color=:red, lw=2)

# Combine them into one figure and save using DrWatson's plotsdir()
p_combined = Plots.plot(p_x, p_y, layout=(2, 1), size=(800, 600))
Plots.savefig(p_combined, plotsdir("velocity_coherence_results.pdf"))

# =========================================
# DIAGNOSTIC PLOT: Check single-trial trace
# =========================================
# Grab a single trial (e.g., the 50th valid trial)

test_idx = 3
sample_trial = valid_trials[test_idx]

# Extract the raw X-coordinate for the left eye
raw_trace = Float64.(sample_trial.left[:, 1])

# Apply detrend function
detrended_trace = remove_offset(raw_trace)

detrended_trace = gradient(detrended_trace, 1/fs)

# Create a time vector in milliseconds for the X-axis
time_ms = (1:length(raw_trace)) 

# Plot the Raw Trace
p_raw = Plots.plot(time_ms, raw_trace, 
    title="Raw Velocity Fixation Trace", 
    xlabel="Time (ms)", 
    ylabel="Screen Coordinate (Pixels/s)", 
    label="Raw Left Eye X", 
    color=:gray, 
    lw=2,
    legend=:topleft)

# Plot the Detrended Trace
p_detrend = Plots.plot(time_ms, detrended_trace, 
    title="Detrended Trace (Ready for FFT)", 
    xlabel="Time (ms)", 
    ylabel="Amplitude (Centered)", 
    label="Detrended", 
    color=:purple, 
    lw=2,
    # xlims=(400,600),
    legend=:topleft)

# Combine them into a 2-panel layout
p_trace_check = Plots.plot(p_raw, p_detrend, layout=(1, 2), size=(900, 400), margin=5Plots.mm)

# Save the diagnostic plot
Plots.savefig(p_trace_check, plotsdir("diagnostic_single_v_trace_3.pdf"))
println("Saved diagnostic trace plot for trial $test_idx.")

mkpath(datadir("processed"))

df_summary = DataFrame(
    Frequency   = frequencies,
    Coherence_X = coherence_x,
    Lower_CI_X  = x_lower_ci,  
    Upper_CI_X  = x_upper_ci,  
    Coherence_Y = coherence_y,
    Lower_CI_Y  = y_lower_ci,  
    Upper_CI_Y  = y_upper_ci   
)
CSV.write(datadir("processed", "summary_v_statistics.csv"), df_summary)

# Save the trial counts and run metadata
df_meta = DataFrame(
    Metric = ["Total_Trials_Raw", "Valid_Trials", "Rejected_Trials", "Rejection_Rate_Pct", "Bootstrap_Iterations", "Sample_Rate", "Time_Halfbandwidth_NW"],
    Value  = [total_count, n_trials, rejected_count, round(rejected_count/total_count * 100, digits=2), n_boot, fs, nw]
)
CSV.write(datadir("processed", "run_v_metadata.csv"), df_meta)

h5open(datadir("processed", "total_coherences.h5"), "w") do h5_file
    h5_file["frequencies"] = frequencies
    h5_file["x_coherence"] = coherence_x
    h5_file["y_coherence"] = coherence_y
    h5_file["x_boot_coherence"] = x_boot_coherence
    h5_file["y_boot_coherence"] = y_boot_coherence
end

h5open(datadir("processed", "v_trials.h5"), "w") do h5_trials_file
    h5_trials_file["x_trials"] = x_trials
    h5_trials_file["y_trials"] = y_trials
    h5_trials_file["fs"] = fs
end

close(file)
