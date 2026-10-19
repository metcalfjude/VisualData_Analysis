using DrWatson
@quickactivate :FractionalSaccades

using Base.Threads
using Statistics
using DataFrames
using LinearAlgebra
using Plots
using CSV
using DSP

include(scriptsdir("calculations", "load_fixational_dynamics.jl"))

# =====================================================================
# FUNCTION DEFINITIONS
# =====================================================================
function zero_phase_filter(trace::AbstractVector, fs=1000; cutoff=150, order=2)
    smoothed = copy(trace)
    valid_mask = .!isnan.(trace)
    sum(valid_mask) < 10 && return smoothed
    
    my_filter = digitalfilter(Lowpass(cutoff, fs=fs), Butterworth(order))
    
    start_idx = 0
    for i in 1:length(trace)
        if valid_mask[i]
            start_idx == 0 && (start_idx = i)
        else
            if start_idx != 0
                segment = trace[start_idx:i-1]
                length(segment) > 15 && (smoothed[start_idx:i-1] .= filtfilt(my_filter, segment))
                start_idx = 0
            end
        end
    end
    if start_idx != 0
        segment = trace[start_idx:end]
        length(segment) > 15 && (smoothed[start_idx:end] .= filtfilt(my_filter, segment))
    end
    return smoothed
end

function local_log_derivative(x, y, window_size=5)
    log_x, log_y = log.(x), log.(y)
    n = length(x)
    slopes = fill(NaN, n)
    for i in 1:(n - window_size + 1)
        idx = i:(i + window_size - 1)
        X = [ones(window_size) log_x[idx]]
        Y = log_y[idx]
        coef = X \ Y
        slopes[i + div(window_size, 2)] = coef[2]
    end
    return slopes
end

# =====================================================================
# MAIN EXECUTION
# =====================================================================
# We use the DURATION setting in load_data to get exactly 1-second chunks 
# anchoring from the start of the fixations.
df = load_data(DURATION=1.0, reject_saccades=false)
valid_df = filter(row -> row.status == :valid, df)

TEST_LIMIT = 1000
if !isnothing(TEST_LIMIT) && TEST_LIMIT < nrow(valid_df)
    valid_df = first(valid_df, TEST_LIMIT)
    println("⚠️ TESTING MODE: Only using the first $TEST_LIMIT valid trials.")
end

num_trials = nrow(valid_df)
println("Total valid event-triggered trials to process: $num_trials")

# =====================================================================
# 1. DEFINE PARAMETERS
# =====================================================================
max_lag = 1000 # Since duration is 1.0s and fs is 1000, max frames = 1000
fs = 1000 
lag_axis_ms = (1:max_lag) ./ fs .* 1000 

qs = collect(0.1:0.1:3.0) 
num_qs = length(qs)

# =====================================================================
# 2. PRE-COMPUTE AND FILTER BINOCULAR GAZE
# =====================================================================
println("Filtering binocular gaze data (150Hz cutoff)...")
mono_gazes = map(eachrow(valid_df)) do row
    raw = (Array(row.left) .+ Array(row.right)) ./ 2.0
    smooth_x = zero_phase_filter(raw[:, 1], fs, cutoff=150)
    smooth_y = zero_phase_filter(raw[:, 2], fs, cutoff=150)
    hcat(smooth_x, smooth_y)
end

# =====================================================================
# 3. CALCULATE PURE ENSEMBLE MOMENTS PER TRIAL
# =====================================================================
println("Calculating Ensemble Moments using $(Threads.nthreads()) threads...")

trial_moments = zeros(Float64, num_trials, max_lag, num_qs)
trial_valid = zeros(Int, num_trials, max_lag)
progress_counter = Threads.Atomic{Int}(0)

Threads.@threads for trial_idx in 1:num_trials
    mono_gaze = mono_gazes[trial_idx]
    trial_len = size(mono_gaze, 1)
    
    # The pure ensemble average anchors completely to the first point (t=0 of the event)
    x0, y0 = mono_gaze[1, 1], mono_gaze[1, 2]
    
    # 'tau' here represents time 't' since the start of the fixation
    for tau in 1:min(max_lag, trial_len - 1)
        x_t, y_t = mono_gaze[tau + 1, 1], mono_gaze[tau + 1, 2]
        
        if !isnan(x0) && !isnan(y0) && !isnan(x_t) && !isnan(y_t)
            d = (x_t - x0)^2 + (y_t - y0)^2
            if d > 0.0 
                for i in 1:num_qs
                    trial_moments[trial_idx, tau, i] = d ^ (qs[i] * 0.5)
                end
                trial_valid[trial_idx, tau] = 1
            end
        end
    end
    
    Threads.atomic_add!(progress_counter, 1)
    (progress_counter[] % 500 == 0) && println("Processed $(progress_counter[]) /$num_trials trials")
end

# Calculate the global ensemble means
moments_matrix = zeros(Float64, max_lag, num_qs)
global_sums = dropdims(sum(trial_moments, dims=1), dims=1)
global_counts = dropdims(sum(trial_valid, dims=1), dims=1)

for tau in 1:max_lag
    if global_counts[tau] > 0
        moments_matrix[tau, :] .= global_sums[tau, :] ./ global_counts[tau]
    else
        moments_matrix[tau, :] .= NaN
    end
end

# =====================================================================
# 4. ENSEMBLE BOOTSTRAPPING (95% CI)
# =====================================================================
B = 200
println("Running $B bootstrap iterations for Ensemble 95% Confidence Intervals...")
boot_rescaled = zeros(Float64, B, max_lag, num_qs)

Threads.@threads for b in 1:B
    sampled_indices = rand(1:num_trials, num_trials)
    
    for tau in 1:max_lag
        c_total = sum(trial_valid[idx, tau] for idx in sampled_indices)
        c_total_1 = sum(trial_valid[idx, 1] for idx in sampled_indices)
        
        if c_total > 0 && c_total_1 > 0
            for i in 1:num_qs
                s_t = sum(trial_moments[idx, tau, i] for idx in sampled_indices)
                s_1 = sum(trial_moments[idx, 1, i] for idx in sampled_indices)
                
                m_t = s_t / c_total
                m_1 = s_1 / c_total_1
                
                # Rescale internally to preserve proper asymmetric bounds
                boot_rescaled[b, tau, i] = (m_t / m_1) ^ (1 / qs[i])
            end
        else
            boot_rescaled[b, tau, :] .= NaN
        end
    end
end

ci_lower = zeros(Float64, max_lag, num_qs)
ci_upper = zeros(Float64, max_lag, num_qs)

for tau in 1:max_lag
    for i in 1:num_qs
        valid_boots = filter(!isnan, boot_rescaled[:, tau, i])
        if !isempty(valid_boots)
            ci_lower[tau, i] = quantile(valid_boots, 0.025)
            ci_upper[tau, i] = quantile(valid_boots, 0.975)
        else
            ci_lower[tau, i] = NaN
            ci_upper[tau, i] = NaN
        end
    end
end

# =====================================================================
# 5. FIND THE SCALING REGIME
# =====================================================================
# For the ensemble average, the plateau might sit at different scales than the time-average.
window_size = 7
local_slopes_q1 = local_log_derivative(lag_axis_ms, moments_matrix[:, findfirst(==(1.0), qs)], window_size)

# Set these after inspecting the derivative plot
fit_start, fit_end = 5, 30 
fit_indices = fit_start:fit_end
trunc_lag = lag_axis_ms[fit_indices]
trunc_moments = moments_matrix[fit_indices, :]

log_X = [ones(length(trunc_lag)) log.(trunc_lag)]
log_Y = log.(trunc_moments)
lambdas = (log_X \ log_Y)[2, :]

# ================ SAVE DATA TO CSV ================s
println("Formatting and saving Ensemble-Averaged data to CSV...")

# 1. Save the time-series moments and confidence intervals
spectrum_df = DataFrame(lag = 1:max_lag, lag_ms = lag_axis_ms)

for i in 1:num_qs
    q = qs[i]
    q_str = replace(string(round(q, digits=1)), "." => "_")
    mu_0 = moments_matrix[1, i]
    
    spectrum_df[!, Symbol("q_", q_str, "_raw_mean")] = moments_matrix[:, i]
    spectrum_df[!, Symbol("q_", q_str, "_rescaled_mean")] = (moments_matrix[:, i] ./ mu_0) .^ (1/q)
    spectrum_df[!, Symbol("q_", q_str, "_rescaled_ci_lower")] = ci_lower[:, i]
    spectrum_df[!, Symbol("q_", q_str, "_rescaled_ci_upper")] = ci_upper[:, i]
end

spectrum_csv = datadir("processed", "ensemble_averaged_spectrum.csv")
CSV.write(spectrum_csv, spectrum_df)

# 2. Save the fitted exponents (Lambdas)
lambda_df = DataFrame(
    q = qs,
    lambda = lambdas,
    fit_start_ms = fill(lag_axis_ms[fit_start], num_qs),
    fit_end_ms = fill(lag_axis_ms[fit_end], num_qs)
)
lambda_csv = datadir("processed", "ensemble_averaged_lambdas.csv")
CSV.write(lambda_csv, lambda_df)

println("Successfully saved ensemble-averaged spectrum and lambdas to $(datadir("processed"))")

# =====================================================================
# 6. VISUAL PRESENTATION
# =====================================================================
moment_plot = plot(
    xlabel="Time t since Fixation Onset (ms)", 
    ylabel="Rescaled Ensemble Moment ⟨|Δx(t)|^q⟩^{1/q}",
    title="Ensemble Averaged Length Scales",
    xscale=:log10,
    yscale=:log10,
    grid=true,
    legend=:outertopright,
    palette=:viridis,
    dpi=300
)

for i in 1:num_qs
    q = qs[i]
    is_labeled = (i % 5 == 1)
    
    mu_0 = moments_matrix[1, i]
    mu = moments_matrix[:, i]
    normalized_moment = (mu ./ mu_0) .^ (1/q)
    
    if is_labeled
        lower_bound = ci_lower[:, i]
        upper_bound = ci_upper[:, i]
        ribbon_tuple = (normalized_moment .- lower_bound, upper_bound .- normalized_moment)
        
        plot!(moment_plot, lag_axis_ms, normalized_moment, 
              label="q = $(round(q, digits=1))", 
              ribbon=ribbon_tuple, fillalpha=0.25,
              linewidth=2.0, alpha=1.0)
    else
        plot!(moment_plot, lag_axis_ms, normalized_moment, 
              label="", 
              linewidth=1.0, alpha=0.3)
    end
end

vspan!(moment_plot, [lag_axis_ms[fit_start], lag_axis_ms[fit_end]], 
       color=:gray, alpha=0.2, label="Fitting Regime")

deriv_plot = plot(
    lag_axis_ms, local_slopes_q1,
    xlabel="Time t since Fixation Onset (ms)",
    ylabel="Local Exponent γ",
    title="Ensemble Logarithmic Derivative (q=1.0)",
    xscale=:log10,
    linewidth=2,
    color=:black,
    xlims=(1, 100),
    ylims=(0, 1.5),
    legend=false,
    dpi=300
)
vspan!(deriv_plot, [lag_axis_ms[fit_start], lag_axis_ms[fit_end]], color=:gray, alpha=0.2)
hline!(deriv_plot, [lambdas[findfirst(==(1.0), qs)]], color=:red, linestyle=:dash)

lambda_plot = plot(
    qs, lambdas, 
    linewidth=3, 
    title = "Ensemble Exponent vs Moment Order (q)", 
    xlabel = "Fractional Moment Order (q)", 
    ylabel = "Exponent λ(q)",
    color=:darkblue,
    legend=:bottomright,
    dpi=300
)

Q_matrix = [ones(length(qs)) qs]
c = Q_matrix \ lambdas
m = round(c[2], digits=3)
α = round(1/c[2], digits=3)

plot!(lambda_plot, qs, (qs .* c[2]) .+ c[1],
    linewidth=2,
    color=:red,
    label = "Linear fit (Slope=$m, α=$α)",
    linestyle=:dash
)

savefig(moment_plot, plotsdir("ensemble_moment_spectrum.png"))
savefig(deriv_plot, plotsdir("ensemble_local_derivative.png"))
savefig(lambda_plot, plotsdir("ensemble_lambda_vs_q_fit.png"))
println("Done! Ensemble-averaged plateau plots saved.")
