%% Task 1 / Implementation of the Himmelblau Function (2 Points)

% See <himmelblau.m>

%% Task 2 / Visualization (2 Points)

[X, Y] = ndgrid(-5:0.1:5, -5:0.1:5);
Z = himmelblau(X, Y);

fig2_1 = figure('Name', 'Figure 2-1: Contour plot');
contour(X, Y, Z, 25);
fig2_2 = figure('Name', 'Figure 2-2: Surface plot');
surf(X, Y, Z);

%% Task 3 / Unconstrained Optimization (2 Points)

himmelblau_Handel = @(a) (a(1)^2 + a(2) - 11)^2 + (a(1) + a(2)^2 - 7)^2;

x{1} = [-3, 3];
x{2} = [-2, 2];
x{3} = [3, 1];
x{4} = [3, -2];

figure(fig2_1)
hold on
for i = 1:4
    opt_point{i} = fminunc(himmelblau_Handel, x{i});
    scatter(opt_point{i}(1), opt_point{i}(2), 'r*')
end
hold off

%% Task 4 / Constrained Optimization (2 Points)

% syms x1 x2
% f = ((x1-5)/2)^2 + (x2-4)^2 -1;
% expand(f);

handle_nonlcon = @nonlcon;

figure(fig2_1)
hold on
for i = 1:4
    opt_point_con{i} = fmincon(himmelblau_Handel, x{i}, [ ], [ ], [ ], [ ], [ ], [ ], handle_nonlcon);
    scatter(opt_point{i}(1), opt_point{i}(2), 'bo')
end
hold off

%% Task 5 / Finding the Local Maximum (1 Point)

x{5} = [-2, 2];
negative_himmelblau = @(x) -himmelblau_Handle(x);

figure(fig2_1)
hold on
opt_point{5} = fminunc(himmelblau_Handel, x{5});
scatter(opt_point{5}(1), opt_point{i}(2), 'cd')
hold off

