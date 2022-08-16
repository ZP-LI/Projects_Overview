ds = datastore('C:\Users\LENOVO\Desktop\Praktikum MATLAB Simulink for Computer Aided Engineering\W2\Data\Data\RKI_COVID19.csv','TreatAsMissing','NA','MissingValue',0)
ds.Delimiter = ',';

ds.VariableNames(3) = {'State'};
ds.VariableNames(7) = {'Cases'};
ds.VariableNames(8) = {'Deaths'};
ds.SelectedVariableNames = {'IdBundesland', 'State', 'Cases', 'Deaths'};
ds.ReadSize = 5000;
sum_cases = 0;
sum_Deaths = 0;

while hasdata(ds)
    T = read(ds);
    sum_cases = sum_cases + sum(T.Cases(strcmp('Schleswig-Holstein', T.State) == 1));
    sum_Deaths = sum_Deaths + sum(T.Deaths(strcmp('Schleswig-Holstein', T.State) == 1));
end
