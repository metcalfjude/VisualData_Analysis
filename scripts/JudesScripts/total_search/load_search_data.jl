using DrWatson
@quickactivate :VisualData_Analysis

using DSP
using TimeseriesTools
using DimensionalData
using JLD2
using KernelDensity
using Plots
using FFTW
using Statistics
using Random
using HDF5
using CSV
using DataFrames

function load_search_data(; max_nan_pct=0.20, pin_first_saccade=false)
    file = jldopen(datadir("native", "hexgrid", "gaze.jld2"), "r")
    index = file["index"]

    subject_groups = DataFrames.groupby(index, :subject_id)
    all_trials = NamedTuple[]
    
    for subj_data in subject_groups
        subject_id = first(subj_data.subject_id)
        println("Processing trials for Subject: $subject_id")
        
        for row in eachrow(subj_data)
            cond_id = row.cond_id
            uid = row.uid
            
            trial = file[row.gaze_file]
            
            if all(isnan.(trial))
                continue 
            end
            
            # Extract the raw time vector from the trial to map indices to exact timestamps
            t_vec = times(trial) 
            
            # Start and end indices logic
            if pin_first_saccade
                try
                    start_idx = minimum(x for x in row.sac_offsets if x > row.idx_fixOff) # If zeroing at first saccade, find the first saccade offset after fixation offset
                catch
                    status = :rejected_no_saccade
                    continue
                end
            else
                start_idx = row.idx_fixOff # else, use the fixation offset index as the start
            end
            end_idx = row.idx_arrayOff
            
            status = :unknown
            r_eye_data = nothing
            l_eye_data = nothing
            
            # Ensure indices are valid and logically ordered
            if start_idx >= 1 && end_idx <= length(t_vec) && start_idx < end_idx
                
                exact_start = t_vec[start_idx]
                exact_end = t_vec[end_idx]
                time_range = 𝑡(exact_start .. exact_end)
                
                try
                    r_eye = trial[time_range, Dim{:Eye}(At(:right))]
                    l_eye = trial[time_range, Dim{:Eye}(At(:left))]
                    
                    # ----------------------------------------------------
                    # Percentage-based NaN filtering over the trial window
                    # ----------------------------------------------------
                    r_nan_pct = count(isnan, r_eye) / length(r_eye)
                    l_nan_pct = count(isnan, l_eye) / length(l_eye)
                    
                    if r_nan_pct <= max_nan_pct && l_nan_pct <= max_nan_pct
                        status = :valid
                        r_eye_data = r_eye
                        l_eye_data = l_eye
                    else
                        status = :rejected_too_many_nans
                    end
                catch
                    status = :rejected_bounds
                end
            else
                status = :rejected_invalid_indices
            end
            
            push!(all_trials, (
                subject_id = subject_id,
                cond_id = cond_id,
                uid = uid,
                start_idx = start_idx,
                end_idx = end_idx,
                status = status,
                right = r_eye_data,
                left = l_eye_data
            ))
        end
    end
    
    return DataFrame(all_trials)
end
