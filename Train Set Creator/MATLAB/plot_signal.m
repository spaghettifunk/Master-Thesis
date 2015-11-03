function plot_signal(x, y, xlabel, ylabel)
    figure
    plot(x, y);
    legend(xlabel, ylabel, 'best');
    axis tight;
end