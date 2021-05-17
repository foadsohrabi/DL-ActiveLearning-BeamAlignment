function posterios = func_compute_posteriors(PI,S,control_plot)
posterios = {};
for l = S:-1:1
    for k = 1:2^l
        if l==S
            posterios{l}{k} = PI(k);
        else
            posterios{l}{k} = posterios{l+1}{2*k-1}+posterios{l+1}{2*k};
        end
    end
end


if control_plot==1
    nodes = [0,repelem(1:2^S-1,2)]; 
    treeplot(nodes);
    [x,y] = treelayout(nodes);
    cnt = 1;
    for l = 1:S
        for k=1:2^l
            cnt = cnt+1;
            text(x(cnt),y(cnt),num2str(posterios{l}{k},'%1.1f'))
        end
    end
end
       