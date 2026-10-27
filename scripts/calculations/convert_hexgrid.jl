using Pickle
using Glob
using DataFrames
using PythonCall
using DimensionalData
using TimeseriesTools
using DrWatson
using JLD2

@quickactivate "VisualData_Analysis"

folder = datadir("raw", "hexgrid")
outfolder = datadir("native", "hexgrid")
files = readdir(glob"*/*/*/*.pkl", folder)
imgfiles = readdir(glob"*/*/*/*.jpg", folder)
img_conds = map(imgfiles) do f
    f = f |> basename |> splitext |> first
    parts = split(f, "-")
    uid = parts[1] |> Meta.parse
    cond_id = parts[3] |> Meta.parse
    uid => cond_id
end |> Dict

function convert_picklement(x)
    thetype = x |> PythonCall.pytype |> string
    if thetype == "<class 'numpy.ndarray'>"
        if string(x.dtype) == "float32"
            newtype = Array{Float32}
        elseif string(x.dtype) == "float64"
            newtype = Array{Float64}
        elseif string(x.dtype) == "uint8"
            newtype = Array{UInt8}
            # elseif string(PythonCall.pytype(x[0][0])) == "<class 'numpy.void'>"
            #     newtype =
            #     # if string(PythonCall.pytype(x[0])) == "<class 'numpy.float64'>"
            #     #     newtype = Array{Float64}
            #     # elseif string(PythonCall.pytype(x[0])) == "<class 'numpy.int64'>"
            #     #     newtype = Array{Int64}
            #     # elseif string(PythonCall.pytype(x[0])) == "<class 'numpy.ndarray'>"
            #     #     return pyconvert(Array, x.tolist())
            #     # else
            #     #     error("Unknown numpy type: $(PythonCall.pytype(x[1]))")
            #     # end
        else
            newtype = Array
        end
    elseif thetype == "<class 'int'>"
        newtype = Int32
    elseif thetype == "<class 'float'>"
        newtype = Float32
    elseif thetype == "<class 'str'>"
        newtype = String
    elseif thetype == "<class 'NoneType'>"
        newtype = Missing
    else
        error("Unknown type: $thetype")
    end
    pyconvert(newtype, x)
end
function read_pickleframe(f)
    data = pyimport("pandas").read_pickle(f)
    columns = pyconvert(Vector, data.columns)
    columns = pyconvert.(String, columns)
    values = pyconvert(Array, data.values)
    data = DataFrame(values, columns)
    data = convert_picklement.(data)
    subject_id = f |> basename |> splitext |> first
    subject_id = filter(isdigit, subject_id) |> Meta.parse
    data.subject_id .= subject_id
    select!(data, Not(:arcadeTrl))
    data.cond_id = getindex.([img_conds], data.uid)
    return data
end

function DrWatson.tag!(f::JLD2.JLDFile)
    D = Dict("" => []) |> tag!
    ks = filter(!=(""), keys(D))
    for k in ks
        f[k] = D[k]
    end
end
function DrWatson.tag!(filename::String)
    jldopen(filename, "w") do f
        tag!(f)
    end
end

data = map(read_pickleframe, files)
filename = joinpath(outfolder, "gaze.jld2")
jldopen(filename, "w") do file
    df = vcat(data...)
    gaze_file = []
    map(eachrow(df)) do x
        ts = x.time |> vec
        pyoffset = 1
        fix_off = isnan(x.idx_fixOff) ? NaN : ts[Int(x.idx_fixOff) - pyoffset]
        array_on = isnan(x.idx_arrayOn) ? NaN : ts[Int(x.idx_arrayOn) - pyoffset]
        img_on = isnan(x.idx_imgOn) ? NaN : ts[Int(x.idx_imgOn) - pyoffset]
        tar_on = isnan(x.idx_tarOn) ? NaN : ts[Int(x.idx_tarOn) - pyoffset]
        reward_time = isnan(x.idx_reward) ? NaN : ts[Int(x.idx_reward) - pyoffset]
        saccade_onsets = ismissing(x.sac_onsets) ? NaN :
                         Int.(filter(!ismissing, x.sac_onsets)) .- pyoffset
        saccade_offsets = ismissing(x.sac_offsets) ? NaN :
                          Int.(filter(!ismissing, x.sac_offsets)) .- pyoffset
        subject_id = x.subject_id
        err = x.error
        uid = x.uid
        cond_id = x.cond_id
        species = x.species
        isCenTar = x.isCenTar

        md = @strdict fix_off array_on img_on tar_on reward_time subject_id cond_id err isCenTar uid saccade_onsets saccade_offsets species
        md = DimensionalData.Metadata(md)
        X = x.eyes'
        X = reshape(X, :, 2, 2)
        X = ToolsArray(X, (𝑡(ts), Var([:x, :y]), Dim{:Eye}([:left, :right])); metadata,
                       name = "Gaze")

        gaze = rectify(X; dims = 𝑡)
        m = md["saccade_onsets"]
        md["saccade_onsets"] = m isa Number && isnan(m) ? NaN : times(gaze)[m]
        m = md["saccade_offsets"]
        md["saccade_offsets"] = m isa Number && isnan(m) ? NaN : times(gaze)[m]
        gaze = rebuild(gaze; metadata = md)

        name = "species=$species/subject_id=$subject_id/uid=$uid"
        push!(gaze_file, name)
        file[name] = gaze
    end
    df.gaze_file = gaze_file
    select!(df, Not(:eyes))
    # select!(df, Not(:sac_onsets))
    # select!(df, Not(:sac_offsets))
    select!(df, Not(:saccadePrediction))
    select!(df, Not(:sac_binary_onset))
    select!(df, Not(:sac_binary_offset))

    file["index"] = df
    tag!(file)
end

begin # * Generate the image gallery
    function matchcond(f, cond)
        c = basename(f)
        c = split(c, '-')[3]
        return Meta.parse(c) == cond
    end
    matchcond(cond) = f -> matchcond(f, cond)
    conds = img_conds |> values |> unique |> sort
    files = map(conds) do c
        f = imgfiles[findfirst(matchcond(c), imgfiles)]
        ff = datadir("native", "hexgrid", "gallery", "$c$(Base.splitext(f)[2])")
        mkpath(dirname(ff))
        cp(f, ff; force = true)
        return ff
    end
    images = DrWatson.load.(files)

    gallery = Dict(conds .=> images)
    tagsave(datadir("native", "hexgrid", "gallery.jld2"), @strdict gallery)
end
