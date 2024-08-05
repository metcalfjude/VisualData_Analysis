function get_spkcount(fr, win, dt)
    win_dt = win / dt
    samp = 0:win_dt:size(fr, 2)
    samp = Int32.(samp)

    lam = zeros(size(fr, 1), length(samp) - 1)

    for ii in 1:length(samp) - 1
        lam[:, ii] .= sum(fr[:, samp[ii] + 1:samp[ii + 1]], dims=2) .* dt
    end

    spk_n = rand.(Poisson.(lam))
    return spk_n
end

function get_firingrate(dist, sig, peak)
    rate = peak .* exp.(-0.5 .* dist .^ 2 ./ sig .^ 2)
    return rate
end

function get_dist(X, loc, hw)
    distx = mod.((X[1, :] .- loc[:, 1] .+ hw), (2 * hw)) .- hw
    disty = mod.((X[2, :] .- loc[:, 2] .+ hw), (2 * hw)) .- hw

    dist = sqrt.(distx .^ 2 .+ disty .^ 2)
    return dist
end

function fHMC_quadratic(p, avg, init_x)
    T = p[:T]
    a = p[:a]
    dt = p[:dt]
    dta = dt ^ (1 / a)
    n = floor(Int, T / dt)
    t = 0:(n - 1) .* dt

    x = zeros(2, avg) .+ init_x
    v = zeros(2, avg)
    ca = gamma(a - 1) / (gamma(a / 2) ^ 2)

    pd = Distributions.Stable(a, 0, p[:gam] ^ (1 / a), 0)

    X = zeros(2, n, avg)

    for i in 1:n
        f = makePotential(x, p)

        dL = rand(pd, (2, avg))

        r = sqrt.(sum(dL .* dL, dims=1))
        th = rand(avg) .* 2 .* π
        g = r .* [cos.(th); sin.(th)]

        vnew = v .+ p[:beta] .* ca .* f .* dt
        xnew = x .+ p[:gam] .* ca .* f .* dt .+ p[:beta] .* v .* dt .+ g .* dta

        x = xnew
        v = vnew
        x = mod.(x .+ π, 2π) .- π
        X[:, i, :] = x
    end

    return X, t
end

function makePotential(x, p)
    f = 0
    hw = 32

    for j in 1:length(p[:total_well_depth])
        distx = mod.((x[1, :] .- p[:total_well_location][j, 1] .+ hw), (2 * hw)) .- hw
        disty = mod.((x[2, :] .- p[:total_well_location][j, 2] .+ hw), (2 * hw)) .- hw

        grad_x = p[:total_well_depth][j] .* 2 .* distx ./ (p[:total_well_radius][j] .^ 2)
        grad_y = p[:total_well_depth][j] .* 2 .* disty ./ (p[:total_well_radius][j] .^ 2)

        mask = distx .^ 2 .+ disty .^ 2 .> p[:total_well_radius][j] .^ 2
        grad_x[mask] .= 0
        grad_y[mask] .= 0

        f .+= [-grad_x; -grad_y]
    end

    return f
end

function dwellingTime(X, p, thre_dist)
    avg = size(X, 3)
    mean_duration = zeros(length(p[:depth]), avg)
    std_duration = zeros(length(p[:depth]), avg)
    total_duration = zeros(length(p[:depth]), avg)

    for j in 1:avg
        for i in 1:size(p[:location], 1)
            loc = p[:location][i, :]
            dist_stim = sqrt.((X[1, :, j] .- loc[1]) .^ 2 .+ (X[2, :, j] .- loc[2]) .^ 2)

            in_stim = [0; (dist_stim .< thre_dist); 0]
            exit_indices = findall(diff(in_stim) .!= 0)

            in_stim_times = (exit_indices[2:2:end] .- exit_indices[1:2:end-1]) .* p[:dt]
            filt_stim_times = in_stim_times[in_stim_times .> 0]
            mean_duration[i, j] = mean(filt_stim_times)
            std_duration[i, j] = std(filt_stim_times)
            total_duration[i, j] = sum(filt_stim_times)
        end
    end

    return mean_duration, std_duration, total_duration
end

function onoff_t(X, p, thre_dist, ctr)
    hw = 32
    dist = get_dist(X, ctr, hw)

    on = [0; (dist .< thre_dist); 0]
    ind = findall(diff(on) .!= 0)
    on_times = (ind[2:2:end] .- ind[1:2:end-1]) .* p[:dt]

    on = [1; (dist .< thre_dist); 1]
    ind = findall(diff(on) .!= 0)
    off_times = (ind[2:2:end] .- ind[1:2:end-1]) .* p[:dt]

    return on_times, off_times
end

function dwellingDist(X, p, thre_dist)
    in_stim_times = []

    for i in 1:size(p[:location], 1)
        loc = p[:location][i, :]
        dist_stim = sqrt.((X[1, :, 1] .- loc[1]) .^ 2 .+ (X[2, :, 1] .- loc[2]) .^ 2)

        in_stim = [0; (dist_stim .< thre_dist); 0]
        exit_indices = findall(diff(in_stim) .!= 0)

        in_stim_times = (exit_indices[2:2:end] .- exit_indices[1:2:end-1]) .* p[:dt]
    end

    return in_stim_times
end
