using DrWatson
@quickactivate :FractionalSaccades

using Base.Threads
using Statistics
using DataFrames
using LinearAlgebra
using Plots
using CSV

include(scriptsdir("JudesScripts/mature_statistics/total_search/load_search_data.jl"))

df = load_search_data()
valid_df = filter(row -> row.status == :valid, df)

# To test the logic on a smaller sample, you can set TEST_LIMIT to the total number of valid trials you want to use.
# Set to 'nothing' to run the full dataset
TEST_LIMIT = 10_000

if !isnothing(TEST_LIMIT) && TEST_LIMIT < nrow(valid_df)
    valid_df = first(valid_df, TEST_LIMIT)
    println("⚠️ TESTING MODE: Only using the first $TEST_LIMIT valid trials.")
end

num_trials = nrow(valid_df)
println("Total valid trials to process: $num_trials")

# =====================================================================
# 1. DEFINE PARAMETERS
# =====================================================================
max_lag = 1000

# Define the spectrum of fractional moments to calculate
# 0.5 = Cauchy test, 1.0 = Mean AD, 2.0 = Mean Squared Displacement
qs = collect(0.1:0.1:3.0) 
num_qs = length(qs)

# =====================================================================
# 2. PRE-COMPUTE BINOCULAR GAZE
# =====================================================================
println("Pre-computing binocular gaze data...")
mono_gazes = map(eachrow(valid_df)) do row
    ( Array(row.left) .+ Array(row.right) ) ./ 2.0
end

# =====================================================================
# 3. CALCULATE SPECTRUM OF MOMENTS (MULTI-THREADED)
# =====================================================================
# Matrix to store results: rows = time lags, cols = q values
moments_matrix = zeros(Float64, max_lag, num_qs)
progress_counter = Threads.Atomic{Int}(0)

println("Calculating Moment Spectrum using $(Threads.nthreads()) threads...")

Threads.@threads for tau in 1:max_lag
    
    # Thread-local accumulator for the q-spectrum to prevent memory allocations
    sum_dists = zeros(Float64, num_qs)
    count_dist = 0
    
    for mono_gaze in mono_gazes
        trial_len = size(mono_gaze, 1)
        
        if tau >= trial_len
            continue 
        end
        
        # Innermost loop: heavily optimized, no array allocations here
        for t in 1:(trial_len - tau)
            x1, y1 = mono_gaze[t, 1], mono_gaze[t, 2]
            x2, y2 = mono_gaze[t + tau, 1], mono_gaze[t + tau, 2]
            
            if !isnan(x1) && !isnan(y1) && !isnan(x2) && !isnan(y2)
                d = (x2 - x1)^2 + (y2 - y1)^2
                
                # Filter out strictly zero displacements (e.g., identical frames)
                # This prevents DOMAIN errors if d=0 is raised to a negative power 
                # or logged later during your exponent fitting.
                if d > 0.0 
                    for i in 1:num_qs
                        sum_dists[i] += d ^ (qs[i] / 2)
                    end
                    count_dist += 1
                end
            end
        end
    end

    # Compute the mean for each q at this specific time lag
    if count_dist > 0
        for i in 1:num_qs
            moments_matrix[tau, i] = sum_dists[i] / count_dist
        end
    else
        for i in 1:num_qs
            moments_matrix[tau, i] = NaN
        end
    end
    
    # Progress tracking
    Threads.atomic_add!(progress_counter, 1)
    current_progress = progress_counter[]
    if current_progress % 100 == 0
        println("Completed $current_progress / $max_lag lags")
    end
end

# =====================================================================
# 4. SAVE TO CSV
# =====================================================================
println("Formatting and saving to CSV...")

# Create a DataFrame. Add the lag time as the first column.
results_df = DataFrame(lag = 1:max_lag)

# Add each q-moment as its own column, with decimal places replaced with underscore (e.g., "q_0.1", "q_1.0")
for i in 1:num_qs
    col_name = Symbol("q_", replace(string(round(qs[i], digits=1)), "." => "_"))
    results_df[!, col_name] = moments_matrix[:, i]
end

csv_path = datadir("processed", "moment_spectrum.csv")
CSV.write(csv_path, results_df)
println("Successfully saved moment spectrum to $csv_path")

# =====================================================================
# 5. QUICK VERIFICATION PLOT
# =====================================================================
fs = 1000 
lag_axis_ms = (1:max_lag) ./ fs .* 1000 

# Plot q=0.5, q=1.0, and q=2.0 as a sanity check
moment_plot = plot(
    xlabel="Time Lag Δt (ms)", 
    ylabel="Fractional Moment <|Δx|^q>",
    title="Spectrum of Displacement Moments",
    xscale=:log10,
    yscale=:log10,
    grid=true,
    legend=:topleft
)

plot!(moment_plot, lag_axis_ms, (results_df[!, :q_0_1] / results_df[1, :q_0_1]) .^ 10, label="q = 0.1 (Low q Test)", linewidth=2)
plot!(moment_plot, lag_axis_ms, (results_df[!, :q_0_5] / results_df[1, :q_0_5]) .^ 2, label="q = 0.5 (Cauchy Test)", linewidth=2)
plot!(moment_plot, lag_axis_ms, (results_df[!, :q_1_0] / results_df[1, :q_1_0]) .^ 1, label="q = 1.0 (MAD)", linewidth=2)
plot!(moment_plot, lag_axis_ms, (results_df[!, :q_2_0] / results_df[1, :q_2_0]) .^ 0.5, label="q = 2.0 (MSD)", linewidth=2)
plot!(moment_plot, lag_axis_ms, (results_df[!, :q_3_0] / results_df[1, :q_3_0]) .^ (1/3), label="q = 3.0 (MCD)", linewidth=2)

trunc_lag = lag_axis_ms[1:30]
trunc = results_df[1:30,:]

# Plot q=0.5, q=1.0, and q=2.0 as a sanity check
truncated_moment_plot = plot(
    xlabel="Time Lag Δt (ms)", 
    ylabel="Fractional Moment <|Δx|^q>",
    title="Spectrum of Displacement Moments",
    xscale=:log10,
    yscale=:log10,
    grid=true,
    legend=:topleft
)

plot!(truncated_moment_plot, trunc_lag, (trunc[!, :q_0_1] / trunc[1, :q_0_1]) .^ 10, label="q = 0.1 (low q Test)", linewidth=2)
plot!(truncated_moment_plot, trunc_lag, (trunc[!, :q_0_5] / trunc[1, :q_0_5]) .^ 2, label="q = 0.5 (Cauchy Test)", linewidth=2)
plot!(truncated_moment_plot, trunc_lag, (trunc[!, :q_1_0] / trunc[1, :q_1_0]) .^ 1, label="q = 1.0 (MAD)", linewidth=2)
plot!(truncated_moment_plot, trunc_lag, (trunc[!, :q_2_0] / trunc[1, :q_2_0]) .^ 0.5, label="q = 2.0 (MSD)", linewidth=2)
plot!(truncated_moment_plot, trunc_lag, (trunc[!, :q_3_0] / trunc[1, :q_3_0]) .^ (1/3), label="q = 3.0 (MCD)", linewidth=2)

x = Matrix(trunc)[:,1]
log_X = [ones(length(x)) log.(x)]

Y = Matrix(trunc)
log_Y = log.(Y[:,2:end])

lambdas = (log_X \ log_Y)[2,:]

lambda_plot = plot(qs, 
    lambdas, 
    linewidth=3, 
    title = "Fitted Exponent against Order (q)", 
    label = "Power Law Exponent", 
    xlabel = "Fractional Moment Order (q)", 
    ylabel = "Exponent"
)

Q = [ones(length(qs)) qs]
Λ = lambdas

c = Q \ Λ

m = round(c[2], digits=3)
α = round(1/c[2], digits=3)

plot!(lambda_plot, qs, (qs .* c[2]) .+ c[1],
    linewidth=4,
    label = "Fitted slope (ν=$m and α=$α)",
    linestyle=:dash
)

savefig(moment_plot, plotsdir("data_moment_spectrum_preview.png"))
savefig(truncated_moment_plot, plotsdir("truncated_data_moment_spectrum_preview.png"))
savefig(lambda_plot, plotsdir("data_lambda_data_fit.png"))
println("Done! Preview plot saved.")
