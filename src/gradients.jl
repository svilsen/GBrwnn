## Gradient for least squares loss function
function gr_ls(y::Vector{Float64}, yhat::Vector{Float64})
    gr = 0.5 .* (y - yhat);
    return gr;
end

## Gradient for least absolute deviation loss function 
function gr_lad(y::Vector{Float64}, yhat::Vector{Float64}) 
    gr = sign.(y - yhat)
    return gr;
end

## Gradient for Huber loss function
function gr_huber(x::Float64, y::Float64, delta::Float64) 
    d_n = x[n] - y[n];
    if abs(d_n) < delta
        return d_n;
    else 
        return delta * sign(d_n);
    end
end

function gr_huber(y::Vector{Float64}, yhat::Vector{Float64}, delta::Float64)
    N = length(y);
    gr = zeros(N);
    for n = 1:N 
        gr[n] = gr_huber(y[n], yhat[n], delta);
    end

    return gr;
end
