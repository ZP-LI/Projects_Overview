function [ z ] = modelFunction(x, y, p)
    % Inputs:
    % - x, y: values of the independent variables
    % - p: parameter vector
    %
    % Outputs:
    % - z: model output
z = - p(1) .* (x - 5).^2 - p(2) .* y.^2 + p(3);

end
