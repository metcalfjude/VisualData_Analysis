function fractional_sampling(varargin)

    savefigpath = './fig/'; % folder for figures
    savedatapath = './data/'; % folder for data

    if not(isfolder(savefigpath))
        mkdir(savefigpath)
    end

    if not(isfolder(savedatapath))
        mkdir(savedatapath);
    end

    %%

    p.location = [0, 0]; % the location of the cued potential well ,
    p.radius = 15; %  the radius of the cued potential well
    p.depth = 50; %  the cue-induced increase of the cued potential well depth

    % the radius of the potential well of stimuli is the same as that of attention/cue

    % Defining walker parameters
    p.a = 1.2; p.gam = 100; p.beta = 1;

    % Defining imulation parameters
    p.dt = 1e-3; p.T = 2e2; avg = 1;

    tri_n = 2; % number of trials
    stim_num = 2; % number of stimuli

    if stim_num == 2
        p.stim_depth = [50 50]; % depth of potential well for stimuli
        p.stim_loc = [0 0; 32 32]; % stimuli potential well locations
    elseif stim_num == 1
        p.stim_depth = [50];
        p.stim_loc = [0 0];
    end

    ifmodu_fr = 1; % whether to increase the baseline local firing rate at the cued position; 1 for True; 0 for False

    tic
    %
    x = -31.5:1:31.5;
    y = -31.5:1:31.5;
    [Yn, Xn] = ndgrid(y, x); %spatial discretization; 64*64 lattice

    latt = zeros(64 * 64, 2); % coordinates of the neurons

    latt(:, 2) = Yn(:);
    latt(:, 1) = Xn(:);

    hw = 32; % half width of the network
    dist_n = get_dist(latt', [0 0], hw); % distance of each neuron to the center of network
    neurang = 5; % neurons within 'neurang' to the center will be analysed
    loc_neu_cho = latt(dist_n <= neurang, :); %  coordinates of the neurons chosen to be analysed
    p.neurang = neurang;

    win_all = [55] / 1000; % spike count window length for Fano factor and noise corraltion

    fano = zeros(length(loc_neu_cho), length(win_all), tri_n); % Fano factor
    mean_spk = zeros(length(loc_neu_cho), length(win_all), tri_n); % mean of spike counts
    var_spk = zeros(length(loc_neu_cho), length(win_all), tri_n); % variance of spike counts
    corr_spk = zeros(length(loc_neu_cho) / 2 * (length(loc_neu_cho) - 1), length(win_all), tri_n); % noise correlation

    %%% fft_len = 5000;
    %%% coef_fft = zeros(tri_n, fft_len);

    peak = 20; % peak firing rate of firing rate gaussian bump
    sig = 12; % sigma of firing rate gaussian bump
    fr_base = 3; % baseline firing rate

    p.peak = peak;
    p.sig = sig;
    p.fr_base = fr_base;
    p.ifmodu_fr = ifmodu_fr;

    %%
    for tri = 1:tri_n

        disp(['Trial: ', num2str(tri)])
        init_x = [0; 0]; % initial location of the pattern
        init_x(1) = rand * 64 - 32;
        init_x(2) = rand * 64 - 32;

        % tic
        % depth, radius, and location of total potential well, i.e. stimuli + attention
        total_well_depth = p.stim_depth;
        total_well_depth(1) = total_well_depth(1) + p.depth;
        total_well_radius = ones(size(total_well_depth)) * p.radius;
        p.total_well_depth = total_well_depth;
        p.total_well_radius = total_well_radius;
        p.total_well_location = p.stim_loc;

        % simulate trajectory of pattern
        [X, t] = fHMC_quadratic(p, avg, init_x);

        % On and Off duration
        [on_times, off_times] = onoff_t(X, p, 5, [0 0]);
        p.on_times{tri} = on_times;
        p.off_times{tri} = off_times;

        % Sampling probability
        hw = 32;
        dist = get_dist(X, [0, 0], hw);
        visit_p = sum((dist < 5)) / numel(dist);
        p.visit_p(tri, 1) = visit_p;

        % find the increase in the baseline firing rate by attention
        dist_neu = get_dist(loc_neu_cho', p.location, hw);
        z = p.depth' .* ((dist_neu .^ 2) ./ (p.radius .^ 2)' - 1);
        % z = p.depth' .* exp(-0.5*dist_neu.^2./(p.radius'.^2));
        z(z > 0) = 0;
        z = sum(z, 1);

        if ifmodu_fr
            frmodu = -z * 0.2;
        else
            frmodu = 0;
        end

        %     size(frmodu)

        % Get instantaneous firing rate for each neuron defined by the firing rate gaussian bump
        dist = get_dist(X, loc_neu_cho, hw);
        fr_gau = get_firingrate(dist, sig, peak);
        % size(fr_gau)

        fr = (fr_gau + fr_base + frmodu'); % total instantaneous firing rate for each neuron

        %   fr_m = mean(fr(:,2001:2000 + fft_len*35), 1); % 18

        transient = 2; % second; transient period that will be excluded from analyses

        % calculate Fano Factor, noise correlation
        for ii = 1:length(win_all)
            spk_n = get_spkcount(fr(:, int32(transient / p.dt) + 1:end), win_all(ii), p.dt);

            corrspk = corrcoef(spk_n');
            corrspk = corrspk(triu(true(size(corrspk)), 1));

            fano(:, ii, tri) = var(spk_n, [], 2) ./ mean(spk_n, 2);
            mean_spk(:, ii, tri) = mean(spk_n, 2);
            var_spk(:, ii, tri) = var(spk_n, [], 2);
            corr_spk(:, ii, tri) = corrspk;

        end

        %     plot the firing rate of an example neuron
        %     if tri == 1
        %         fig1 = figure;
        %         subplot(1,1,1)
        %         plot(t(2000:4000), fr(1, 2000:4000),'DisplayName','rate')
        %         hold on
        %         plot(t(2000:4000), dist(1, 2000:4000),'DisplayName','dist')
        %         legend()
        %         title([num2str(loc_neu_cho(1,1)),'  ',num2str(loc_neu_cho(1,2))])
        %
        %
        % %         sgtitle(title_fig)
        % %         figname = sprintf('rate_%s.png', title_fig);
        % %         %saveas(fig1, [savefigpath,figname])

    end

    %%
    p.fano = fano;
    p.corr_spk = corr_spk;
    p.mean_spk = mean_spk;
    p.var_spk = var_spk;
    p.win_all = win_all;
    % p.mean_r = mean_r;

    %%

    save([savedatapath, 'data_fano_corr.mat'], 'p')

    % make video for the trajectory of CoM of the random walker
    figure('color', 'w');
    ii = 0;

    for i = 2000:1:4000 %length(t)
        ii = ii + 1;
        %plot([1 -1]*p.location, [0 0],'*k')
        %hold on
        plot(X(1, i), X(2, i), 'ob')
        hold off

        xlabel('x')
        ylabel('y')
        title(['t = ' num2str(t(i))])
        axis([-32 32 -32 32])
        %axis equal
        %pause(1/60)
        movieframe(ii) = getframe(gcf);
    end

    moviname = sprintf('fano_%s.avi', "nothing");

    mywriter = VideoWriter([savefigpath, moviname]);
    mywriter.FrameRate = 20;
    open(mywriter)
    writeVideo(mywriter, movieframe)
    close(mywriter)

end

%% Functions

function spk_n = get_spkcount(fr, win, dt)
    win_dt = win / dt;
    %disp(win_dt)
    samp = 0:win_dt:size(fr, 2);
    % samp(1:10)
    samp = int32(samp);
    % samp(1:10)

    %spk_n = zeros(size(samp));
    lam = zeros(size(fr, 1), size(samp, 2));

    for ii = 1:length(samp) - 1

        lam(:, ii) = sum(fr(:, samp(ii) + 1:samp(ii + 1)), 2) * dt;
        %     spk_n(ii) = poissrnd(lam);

    end

    spk_n = poissrnd(lam(:, 1:end - 1));

end

%%

function [rate] = get_firingrate(dist, sig, peak)

    rate = peak * exp(-0.5 * dist .^ 2 / sig .^ 2);

end

%%
function [dist] = get_dist(X, loc, hw)

    distx = mod((X(1, :) - loc(:, 1) + hw), (2 * hw)) - hw;
    disty = mod((X(2, :) - loc(:, 2) + hw), (2 * hw)) - hw;

    dist = sqrt(distx .^ 2 + disty .^ 2);

end

function [X, t] = fHMC_quadratic(p, avg, init_x)

    %     p.radius = radius;
    %     p.depth = depth;
    T = p.T;
    a = p.a;
    dt = p.dt; %1e-3; %integration time step (s)
    dta = dt .^ (1 / a); %fractional integration step
    n = floor(T / dt); %number of samples
    t = (0:n - 1) * dt;

    x = zeros(2, avg) + init_x; %[0;0]; %initial condition for each parallel sim
    v = zeros(2, avg); %+[1;1];
    ca = gamma(a - 1) / (gamma(a / 2) .^ 2);

    pd = makedist('Stable', 'alpha', a, 'gam', p.gam ^ (1 / a));

    X = zeros(2, n, avg);

    for i = 1:n
        f = makePotential(x, p); % x ca for fractional derivative

        %         dL = stblrnd(a,0,p.gam,0,[2,avg]);
        dL = pd.random([2, avg]);

        r = sqrt(sum(dL .* dL, 1)); %step length

        th = rand(1, avg) * 2 * pi;
        g = r .* [cos(th); sin(th)];

        % Stochastic fractional Hamiltonian Monte Carlo
        vnew = v + p.beta * ca * f * dt;
        xnew = x + p.gam * ca * f * dt + p.beta * v * dt + g * dta;

        x = xnew;
        v = vnew;
        % x = wrapToPi(x); % apply periodic boundary to avoid run-away
        x = wrapToPi(x .* pi ./ 32) .* 32 ./ pi;
        X(:, i, :) = x;
    end

end

function f = makePotential(x, p)
    % Just make the gradient 0 past a certain radius.
    f = 0;
    hw = 32;

    for j = 1:length(p.total_well_depth)

        distx = mod((x(1, :) - p.total_well_location(j, 1) + hw), (2 * hw)) - hw;
        disty = mod((x(2, :) - p.total_well_location(j, 2) + hw), (2 * hw)) - hw;

        %grad_x = p.total_well_depth(j) * exp(-0.5*(distx.^2+disty.^2)/(p.total_well_radius(j).^2)) * distx/(p.total_well_radius(j).^2);
        %grad_y = p.total_well_depth(j) * exp(-0.5*(distx.^2+disty.^2)/(p.total_well_radius(j).^2)) * disty/(p.total_well_radius(j).^2);

        grad_x = p.total_well_depth(j) * 2 * distx ./ (p.total_well_radius(j) .^ 2);
        grad_y = p.total_well_depth(j) * 2 * disty ./ (p.total_well_radius(j) .^ 2);

        grad_x(distx .^ 2 + disty .^ 2 > p.total_well_radius(j) .^ 2) = 0;
        grad_y(distx .^ 2 + disty .^ 2 > p.total_well_radius(j) .^ 2) = 0;

        f = f + [-grad_x; -grad_y];

    end

end

function [mean_duration, std_duration, total_duration] = dwellingTime(X, p, thre_dist)
    % Find mean duration the sampling point stays in a well
    avg = size(X, 3);
    mean_duration = zeros(length(p.depth), avg);
    std_duration = zeros(length(p.depth), avg);
    total_duration = zeros(length(p.depth), avg);

    for j = 1:avg

        for i = 1:size(p.location, 1)
            loc = p.location(i, :);
            dist_stim = sqrt((X(1, :, j) - loc(1)) .^ 2 + (X(2, :, j) - loc(2)) .^ 2);

            %             in_stim = [0,(dist_stim < sqrt(p.radius(i))),0];
            in_stim = [0, (dist_stim < thre_dist), 0];
            exit_indices = find([false, in_stim] ~= [in_stim, false]);

            in_stim_times = (exit_indices(2:2:end) - exit_indices(1:2:end - 1)) * p.dt;
            filt_stim_times = in_stim_times(in_stim_times > 0);
            mean_duration(i, j) = mean(filt_stim_times);
            std_duration(i, j) = std(filt_stim_times);
            total_duration(i, j) = sum(filt_stim_times);
        end

    end

end

function [on_times, off_times] = onoff_t(X, p, thre_dist, ctr)
    hw = 32;
    dist = get_dist(X, ctr, hw);

    on = [0, (dist < thre_dist), 0];
    ind = find(diff(on) ~= 0);
    on_times = (ind(2:2:end) - ind(1:2:end - 1)) * p.dt;

    on = [1, (dist < thre_dist), 1];
    ind = find(diff(on) ~= 0);
    off_times = (ind(2:2:end) - ind(1:2:end - 1)) * p.dt;

end

function in_stim_times = dwellingDist(X, p, thre_dist)

    for i = 1:size(p.location, 1)
        loc = p.location(i, :);
        dist_stim = sqrt((X(1, :, 1) - loc(1)) .^ 2 + (X(2, :, 1) - loc(2)) .^ 2);

        %         in_stim = [0,(dist_stim < sqrt(p.radius(i))),0];
        in_stim = [0, (dist_stim < thre_dist), 0];

        exit_indices = find([false, in_stim] ~= [in_stim, false]);

        in_stim_times = (exit_indices(2:2:end) - exit_indices(1:2:end - 1)) * p.dt;
    end

end
