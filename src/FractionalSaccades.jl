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
    end
end
macro preamble()
    _preamble()
end

end
