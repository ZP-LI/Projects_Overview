%% Task 1 / Data Import and Preprocessing (1 Point)

load carbig.mat

cyl4 = string(cyl4);
Mfg = string(Mfg);
Model = string(Model);
org = string(org);
Origin = string(Origin);
when = string(when);

%% Task 2 / Basics (3 Points)

Horsepower_min = min(Horsepower);
[Hor_row, Hor_col] = find(Horsepower_min == Horsepower);
Mfg_min = Mfg(Hor_row)
Model_min = Model(Hor_row)

Mfg_uni = unique(Mfg);
for i = 1:length(Mfg_uni)
    Mfg_row = 0;
    Mfg_col = 0;
    [Mfg_row, Mfg_col] = find(Mfg == Mfg_uni(i));
    Weight_median(i) = median(Weight(Mfg_row));
end
Weight_median = [Mfg_uni Weight_median']

%% Task 3 / Boxplots (2 Points)

for i = 1:2
    Mfg_row = 0;
    Mfg_col = 0;
    [Mfg_row, Mfg_col] = find(Mfg == Mfg_uni(i));
    MPG_box = MPG(Mfg_row);
    figure('Name', 'MPG_boxplot')
    boxplot(MPG_box)
end

%% Task 4 / Distribution Fitting (2 Points)

MPG_fit_Norm = fitdist(MPG, 'Normal');
MPG_fit_Weibull = fitdist(MPG, 'Weibull');
nll_Norm = negloglik(MPG_fit_Norm);
nll_Weibull = negloglik(MPG_fit_Weibull);
if nll_Norm > nll_Weibull
    display('nll_Norm fits the data better!')
else
    display('nll_Weibull fits the data better!')
end