function y=NN_impl(prob_x,weights,bias)
a=prob_x.*weights;
y=heaviside(a(1)+a(2)+bias);