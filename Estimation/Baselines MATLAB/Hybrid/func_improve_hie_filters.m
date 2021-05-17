function w_hyb_temp = func_improve_hie_filters(g,N,A_BS,wd_temp)

w_hyb_temp = exp(1j*angle(wd_temp));
w_hyb_temp = w_hyb_temp*(sqrt(1)/norm(w_hyb_temp));

A = A_BS';
b = g;
G = A'*A;
c = A'*b;

x = w_hyb_temp;

for cnt =1:2000
    for i=1:64
        G_row_col_i_removed = G;
        G_row_col_i_removed(i,:) = [];
        G_row_col_i_removed(:,i) = [];
        x_i = x(i);
        x_i_removed = x;
        x_i_removed(i) = [];
        g_ii = G(i,i);
        g_i_removed = G(i,:);
        g_i_removed(i) = [];
        c_i = c(i);
        c_i_removed = c;
        c_i_removed(i) = [];
        
        x_i = (1/sqrt(N))*exp(1j*(angle((g_i_removed*x_i_removed - c_i))+pi));
        x(i) = x_i;        
    end
end


w_hyb_temp = x;
