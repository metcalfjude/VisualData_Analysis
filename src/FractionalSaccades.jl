module FractionalSaccades

if !haskey(ENV, "DRWATSON_STOREPATCH")
    ENV["DRWATSON_STOREPATCH"] = "true"
end

function _preamble()
    quote
        using Statistics
        using StatsBase
        using Random
        using LinearAlgebra
        using FileIO
        using CairoMakie
        using DataFrames
        using DimensionalData
        using DrWatson
        using DSP
        using Foresight
        using IntervalSets
        using Normalization
        using Distributed
        using JLD2
        using HypothesisTests
        using MultipleTesting
        using TimeseriesTools
        using TimeseriesTools.Unitful
        import TimeseriesTools.TimeSeries
        using UnPack
        import Foresight.clip
        import CairoMakie.save
        import DimensionalData: metadata
        using Term
        using Term.Progress
        using Polynomials
    end
end
macro preamble()
    _preamble()
end

export remove_offset, polynomial_detrend

function remove_offset(v::V)::Vector{Float64} where {V<:AbstractVector{Float64}}
    data_len = length(v)
    linear_vec = 1:data_len
    Linear_Matrix = hcat(linear_vec, ones(data_len))

    m, c = Linear_Matrix \ v

    v = @. (v - c) - m * linear_vec
    return v
end

using Polynomials
function polynomial_detrend(v::V, order::Int)::Vector{Float64} where {V<:AbstractVector{Float64}}
    data_len = length(v)
    linear_vec = 1:data_len

    poly_fit = Polynomials.fit(linear_vec, v, order)

    v_new = v .- poly_fit.(linear_vec)

    return v_new
end



end
