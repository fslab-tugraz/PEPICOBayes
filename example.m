function example()
    % load example data
    load('example_data.mat')
    
    %%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%
    
    % Monte Carlo variables
    Nsweep        = 1e4;  % Number of sweeps in MCMC
    Nrun          = 1e1;  % Number of runs per sweep
    cutoff        = 2e3;  % Therminalization
    
    % Step sizes for Monte Carlo algorithm
    sigma_la = 1/5e2;
    sigma_lb = 1/4e2;
    sigma_sigma1 = 1e-2;
    sigma_sigma2 = 1e-2;
    sigma_xi = 1/5e2;
    sigma_xe = 1/5e2;
    sigma_qa = 1/(70*numel(E));
    sigma_qb = 1/(70*numel(E));
    
    % start values
    lambda1 = 0.2;
    lambda2 = 0.6;
    sigma1 = 1e-4;
    sigma2 = 1e-4;
    xi_e = 0.75;
    xi_i = 0.75;
    Pi0 = [lambda1, lambda2, sigma1, sigma2, xi_i, xi_e];
    
    %%%%%%%%%%%%%%%%%%%%% COMPUTATION %%%%%%%%%%%%%%%%%%%%%
    
    n11_beta  = sum(spec_beta(:));
    n11_alpha = sum(spec_alpha(:));
    
    parameter = [Pi0, sigma_qa, sigma_qb, sigma_la, sigma_lb, sigma_sigma1, sigma_sigma2, sigma_xi, sigma_xe, ...
        N_beta,  n00_beta,  n01_beta,  n02_beta,  n03_beta,  n10_beta,  n11_beta,  n12_beta,  n20_beta,  n21_beta,  n30_beta, ...
        N_alpha, n00_alpha, n01_alpha, n02_alpha, n03_alpha, n10_alpha, n11_alpha, n12_alpha, n20_alpha, n21_alpha, n30_alpha,];

    % Monte Carlo Algorithm
    [q1_timeseries, q2_timeseries, Pi, p_acc] = PEPICOBayes(spec_beta, spec_alpha, Nrun, Nsweep, parameter);

    % compute mean value and standard error (thermalize)
    q2_mean = mean(q2_timeseries(:,:, cutoff:end), 3);
    q2_std = std(q2_timeseries(:,:, cutoff:end), 0, 3);
    
    fprintf('Acceptance probability:\n lambda_1: \t%.1f %%\n lambda_2: \t%.1f %%\n sigma_1: \t%.1f %%\n sigma_2: \t%.1f %%\n xi_e: \t\t%.1f %%\n xi_i: \t\t%.1f %%\n q1: \t\t%.1f %%\n q2: \t\t%.1f %%\n', p_acc*100)
    
    
    if ~exist('q2')
        q2 = nan(size(E));
    end
    %%%%%%%%%%%%%%%%%%%%% PLOT RESULTS %%%%%%%%%%%%%%%%%%%%%
    createFigures(E, q2_mean, q2_std, q2, q2_timeseries, Pi)
end

function createFigures(E, q2, q2_std, q2_real, q2_timeseries, Pi)
    cm = lines(size(E,1));
    
    %%%%%%%%% Spectrum Plot %%%%%%%%%
    legend_text = cell(2*size(E,1),1);
    
    figure('Name', 'Spectra')
    set(gcf,'defaultTextInterpreter', 'latex')
    hold on
    for i = 1:size(E, 1)
        errorbar(E(i,:), q2(i,:), q2_std(i,:), 'Color', cm(i,:), 'LineWidth', 2);
        plot(E(i,:), q2_real(i,:), 'Color', cm(i,:)*0, 'LineStyle', '--', 'LineWidth', 2)
        
        legend_text{2*i-1} = ['Molecule ', num2str(i), ' reconstructed'];
        legend_text{2*i} = ['Molecule ', num2str(i), ' real'];
    end
    ylabel('$q_{ij}$')
    xlabel('E')
    hold off
    legend(legend_text, 'Location', 'NorthWest')
    
    %%%%%%%%% Timeseries q2 %%%%%%%%%
    figure('Name', 'Timeseries q2_i')
    set(gcf,'defaultTextInterpreter', 'latex')
    for j = 1:size(E, 1)
        for i = 1:min(size(E,2),6)
            subplot(min(size(E,2),6), size(E,1), (i-1)*size(E, 1)+j)
            plot(squeeze(q2_timeseries(j, i, :)), 'Color', cm(j,:))
            ylabel(['$q_{',num2str(j),',',num2str(i),'}$'], 'FontSize',15)
            pow = floor(log10(min(q2_timeseries(j,i,2000:end))));
            ylim([floor(min(q2_timeseries(j,i,2000:end))*10^(-pow))*10^pow, ceil(min(q2_timeseries(j,i,2000:end))*10^(-pow))*10^pow])
        end
        xlabel('k', 'FontSize',15)
    end
    
    %%%%%%%%% Timeseries Parameter %%%%%%%%%
    ylabels = {'$\underline{\lambda}_1$', '$\underline{\lambda}_2$', '$\sigma_1$', '$\sigma_2$', '$\xi_i$', '$\xi_e$'};
    figure('Name', 'Timeseries Parameter')
    set(gcf,'defaultTextInterpreter', 'latex')
    for i = 1:6
        subplot(6,1,i)
        plot(Pi(i,:))
        ylabel(ylabels{i}, 'FontSize',15)
        
        if i == 3 || i == 4
            set(gca,'YScale', 'log')
        end
    end
    xlabel('k', 'FontSize',15)
end