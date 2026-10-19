using DrWatson
@quickactivate :FractionalSaccades

using Base.Threads
using Statistics
using DataFrames
using LinearAlgebra
using Plots

include(scriptsdir("JudesScripts/mature_statistics/total_search/load_search_data.jl"))

df = load_search_data(pin_first_saccade=false)
valid_df = filter(row -> row.status == :valid, df)

# To test the logic on a smaller sample, you can set TEST_LIMIT to the total number of valid trials you want to use.
# Set to 'nothing' to run the full dataset
TEST_LIMIT = 1000

if !isnothing(TEST_LIMIT) && TEST_LIMIT < nrow(valid_df)
    valid_df = first(valid_df, TEST_LIMIT)
    println("⚠️ TESTING MODE: Only using the first $TEST_LIMIT valid trials.")
end

num_trials = nrow(valid_df)
println("Total valid trials to process: $num_trials")

max_lag = 1000

# =====================================================================
# 2. PRE-COMPUTE BINOCULAR GAZE
# =====================================================================
# We will iterate over the trials multiple times (once per lag), 
# so we calculate the binocular average once upfront to save CPU time.
println("Pre-computing binocular gaze data...")
bino_gazes = map(eachrow(valid_df)) do row
    (Array(row.right) .+ Array(row.left)) ./ 2.0
end

# =====================================================================
# 3. CALCULATE CURVE (MULTI-THREADED)
# =====================================================================
# Rename output variable to reflect Mean AD
time_avg_mean_ad_curve = zeros(Float64, max_lag)

progress_counter = Threads.Atomic{Int}(0)

println("Calculating Time-Averaged Mean AD Curve using $(Threads.nthreads()) threads...")

Threads.@threads for tau in 1:max_lag
    
    # Instead of an array, use scalar counters for zero memory allocations
    sum_dist = 0.0
    count_dist = 0
    
    for bino_gaze in bino_gazes
        trial_len = size(bino_gaze, 1)
        
        if tau >= trial_len
            continue 
        end
        
        for t in 1:(trial_len - tau)
            x1, y1 = bino_gaze[t, 1], bino_gaze[t, 2]
            x2, y2 = bino_gaze[t + tau, 1], bino_gaze[t + tau, 2]
            
            if !isnan(x1) && !isnan(y1) && !isnan(x2) && !isnan(y2)
                d = sqrt((x2 - x1)^2 + (y2 - y1)^2)
                
                # Accumulate sum and count directly
                sum_dist += d
                count_dist += 1
            end
        end
    end
    
    # Compute the mean directly
    if count_dist == 0
        time_avg_mean_ad_curve[tau] = NaN
    else
        time_avg_mean_ad_curve[tau] = sum_dist / count_dist
    end
    
    # Progress tracking
    Threads.atomic_add!(progress_counter, 1)
    current_progress = progress_counter[]
    if current_progress % 100 == 0
        println("Completed $current_progress / $max_lag lags")
    end
end

# =====================================================================
# 4. PLOTTING
# =====================================================================
fs = 1000 
lag_axis_ms = (1:max_lag) ./ fs .* 1000 

x_trunc = lag_axis_ms[1:100]
y_trunc = time_avg_mean_ad_curve[1:100]
log_X = [ones(length(x_trunc)) log10.(x_trunc)]
log_Y = log10.(y_trunc)

c = log_X \ log_Y
m = round(c[2], digits=2)
α = round(3-2*c[2], digits=2)

mad_plot = plot(lag_axis_ms[2:end], time_avg_mean_ad_curve[2:end], 
    xlabel="Time Lag Δt (ms)", 
    ylabel="Time-Averaged Mean AD",
    title="Time-Averaged Mean AD Curve",
    label="Population Mean",
    xscale=:log10,
    yscale=:log10,
    linewidth=3,
    color=:blue,
    legend=:topleft,
    grid=true)

plot!(lag_axis_ms[2:end], ((lag_axis_ms[2:end] ) .^ c[2]) .* 10^(c[1]),
    label = "Best Fit (m=$m α=$α)",
    linewidth=3,
    linestyle=:dash,
    color=:black
)
# plot!(lag_axis_ms[2:end], (lag_axis_ms[2:end] .* 0.1).^ 1.5)

savefig(mad_plot, plotsdir("time_avg_mean_ad_curve.png"))
println("Done! Plot saved.")
