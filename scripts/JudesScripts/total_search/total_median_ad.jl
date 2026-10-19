using DrWatson
@quickactivate :FractionalSaccades

using Base.Threads
using Statistics
using DataFrames
using LinearAlgebra
using Plots

include(scriptsdir("JudesScripts/mature_statistics/total_search/load_search_data.jl"))

df = load_search_data()
valid_df = filter(row -> row.status == :valid, df)

# To test the logic on a smaller sample, you can set TEST_LIMIT to the total number of valid trials you want to use.
TEST_LIMIT = 10 # Set to 'nothing' to run the full dataset

if !isnothing(TEST_LIMIT) && TEST_LIMIT < nrow(valid_df)
    valid_df = first(valid_df, TEST_LIMIT)
    println("⚠️ TESTING MODE: Only using the first $TEST_LIMIT valid trials.")
end

num_trials = nrow(valid_df)
println("Total valid trials to process: $num_trials")

max_lag = 10000

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
time_avg_mad_curve = zeros(Float64, max_lag)

# Create a thread-safe counter for progress updates
progress_counter = Threads.Atomic{Int}(0)

println("Calculating Time-Averaged MAD Curve using $(Threads.nthreads()) threads...")

# The @threads macro distributes the lags across available CPU cores
Threads.@threads for tau in 1:max_lag
    
    # tau_dists is created INSIDE the loop. 
    # This ensures each thread gets its own independent array (no race conditions)
    tau_dists = Float64[]
    
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
                push!(tau_dists, d)
            end
        end
    end
    
    # Writing to a pre-allocated array is thread-safe as long as 
    # each thread writes to a unique index (tau)
    if isempty(tau_dists)
        time_avg_mad_curve[tau] = NaN
    else
        time_avg_mad_curve[tau] = median(tau_dists)
    end
    
    # Safely update and read the progress counter across multiple threads
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

mad_plot = plot(lag_axis_ms, time_avg_mad_curve, 
    xlabel="Time Lag Δt (ms)", 
    ylabel="Time-Averaged Median AD",
    title="Time-Averaged MAD Curve",
    label="Population Median",
    xscale=:log10,
    yscale=:log10,
    linewidth=2,
    color=:blue,
    grid=true)

savefig(mad_plot, plotsdir("time_avg_median_ad_curve.png"))
println("Done! Plot saved.")
