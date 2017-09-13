function example()
    % load example data
    load('example_data.mat')

    %%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%
    
    % Monte Carlo variables
    Nsweep        = 1e4;  % Number of sweeps in MCMC
    Nrun          = 1e2;  % Number of runs per sweep
    cutoff        = 1e3;  % Therminalization
    
    % Step sizes for Monte Carlo algorithm
    sigma_la = 1/5e2;
    sigma_lb = 1/5e2;
    sigma_xi = 1/5e2;
    sigma_xe = 1/5e2;
    sigma_qa = 1/(70*numel(E));
    sigma_qb = 1/(70*numel(E));
    
    % start values
    lambda1 = 0.2;
    lambda2 = 0.6;
    xi_e = 0.75;
    xi_i = 0.75;
    Pi0 = [lambda1, lambda2, xi_i, xi_e];
    
    parameter = [sigma_qa, sigma_qb, sigma_la, sigma_lb, sigma_xi, sigma_xe, ...
        Np_beta, N00_beta, N01_beta, N02_beta, N03_beta, N10_beta, N11_beta, N12_beta, N20_beta, N21_beta, N30_beta, ...
        Np_alpha, N00_alpha, N01_alpha, N02_alpha, N03_alpha, N10_alpha, N11_alpha, N12_alpha, N20_alpha, N21_alpha, N30_alpha,];

    %%%%%%%%%%%%%%%%%%%%% COMPUTATION %%%%%%%%%%%%%%%%%%%%%
    
    % Monte Carlo Algorithm
    [q1_timeseries, q2_timeseries, Pi, p_acc] = PEPICOBayes(n_beta, n_alpha, Nrun, Nsweep, parameter, Pi0);

    % compute mean value and standard error (thermalize)
    q2 = mean(q2_timeseries(:,:, cutoff:end), 3);
    q2_std = std(q2_timeseries(:,:, cutoff:end), 0, 3);
    
    fprintf('Acceptance probability:\n lambda_1: \t%.1f %%\n lambda_2: \t%.1f %%\n xi_e: \t\t%.1f %%\n xi_i: \t\t%.1f %%\n q1: \t\t%.1f %%\n q2: \t\t%.1f %%\n', p_acc*100)
    
    %%%%%%%%%%%%%%%%%%%%% PLOT RESULTS %%%%%%%%%%%%%%%%%%%%%
    createFigures(E, q2, q2_std, q2_real, q2_timeseries, Pi)
end

function createFigures(E, q2, q2_std, q2_real, q2_timeseries, Pi)
    cm = lines(size(E,1));
    
    %%%%%%%%% Spectrum Plot %%%%%%%%%
    legend_text = cell(2*size(E,1),1);
    
    figure('Name', 'Spectra')
    hold on
    for i = 1:size(E, 1)
        plot(E(i,:), q2_real(i,:), 'Color', cm(i,:), 'LineStyle', '--', 'LineWidth', 2)
        errorbar(E(i,:), q2(i,:), q2_std(i,:), 'Color', cm(i,:), 'LineWidth', 2);
        
        legend_text{2*i-1} = ['Molecule ', num2str(i), ' reconstructed'];
        legend_text{2*i} = ['Molecule ', num2str(i), ' real'];
    end
    ylabel('q_{ij}')
    xlabel('E')
    hold off
    legend(legend_text, 'Location', 'NorthWest')

    %%%%%%%%% Timeseries q2 %%%%%%%%%
    figure('Name', 'Timeseries q2_i')
    for j = 1:size(E, 1)
        for i = 1:size(E,2)
            subplot(size(E,2), size(E,1), (i-1)*size(E, 1)+j)
            plot(squeeze(q2_timeseries(j, i, :)), 'Color', cm(j,:))
            ylabel(['q_{',num2str(j),',',num2str(i),'}'])
        end
        xlabel('k')
    end
    
    %%%%%%%%% Timeseries Parameter %%%%%%%%%
    figure('Name', 'Timeseries Parameter')
    subplot(4,1,1)
    plot(Pi(1,:))
    ylabel('\lambda_1')
    
    subplot(4,1,2)
    plot(Pi(2,:))
    ylabel('\lambda_2')
    
    subplot(4,1,3)
    plot(Pi(3,:))
    ylabel('\xi_e')
    
    subplot(4,1,4)
    plot(Pi(4,:))
    ylabel('\xi_i')
    xlabel('k')
end