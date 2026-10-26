using DrWatson
@quickactivate :FractionalSaccades

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

function load_data(; THRESHOLD=1., BUFFER=0.01, max_speed=3., fs_file=1000)
    file = jldopen(datadir("native", "hexgrid", "gaze.jld2"), "r")
    index = file["index"]

    subject_groups = DataFrames.groupby(index, :subject_id)
    all_fixations = NamedTuple[]
    
    for subj_data in subject_groups
        subject_id = first(subj_data.subject_id)
        println("Processing fixations for Subject: $subject_id")
        
        for row in eachrow(subj_data)
            cond_id = row.cond_id
            uid = row.uid
            
            trial = file[row.gaze_file]
            
            if all(isnan.(trial))
                continue 
            end
            
            meta = DimensionalData.metadata(trial)
            onsets = meta["saccade_onsets"]
            offsets = meta["saccade_offsets"]
            
            # Extract the raw time vector from the trial to find exact indices
            t_vec = times(trial) 
            
            for i in 1:(length(onsets) - 1)
                fix_start = offsets[i]
                fix_end = onsets[i+1]
                dur = fix_end - fix_start
                
                status = :unknown
                r_eye_data = nothing
                l_eye_data = nothing
                
                if dur >= THRESHOLD + 2 * BUFFER
                    ######################
                    ### Indexing logic ###
                    ######################
                    target_start_time = fix_start + BUFFER
                    target_end_time = fix_end - BUFFER

                    start_idx = searchsortedfirst(t_vec, target_start_time)
                    end_idx = searchsortedlast(t_vec, target_end_time)
                    
                    # Ensure we haven't requested an index before the start of the trial
                    if start_idx <= end_idx && end_idx <= length(t_vec)
                        
                        # Map the buffered time bounds to the available sample indices.
                        exact_start = t_vec[start_idx]
                        exact_end = t_vec[end_idx]
                        time_range = 𝑡(exact_start .. exact_end)
                        # ---------------------------------
                        
                        try
                            r_eye = trial[time_range, Dim{:Eye}(At(:right))]
                            l_eye = trial[time_range, Dim{:Eye}(At(:left))]
                            
                            if !any(isnan, r_eye) && !any(isnan, l_eye)
                                r_speed = diff(r_eye[:,1].data).^2 + diff(r_eye[:,2].data).^2
                                l_speed = diff(l_eye[:,1].data).^2 + diff(l_eye[:,2].data).^2
                                
                                if maximum(r_speed) < max_speed && maximum(l_speed) < max_speed
                                    status = :valid
                                    r_eye_data = r_eye
                                    l_eye_data = l_eye
                                else
                                    status = :rejected_micro_saccade
                                end
                            else
                                status = :rejected_nan_or_short
                            end
                        catch
                            status = :rejected_bounds
                        end
                    else
                        status = :rejected_bounds
                    end
                else
                    status = :rejected_duration
                end
                
                push!(all_fixations, (
                    subject_id = subject_id,
                    cond_id = cond_id,
                    uid = uid,
                    fix_start = fix_start,
                    fix_end = fix_end,
                    duration = dur,
                    status = status,
                    right = r_eye_data,
                    left = l_eye_data
                ))
            end
        end
    end
    
    return DataFrame(all_fixations)
end
