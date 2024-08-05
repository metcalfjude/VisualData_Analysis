using FractionalSaccades
FractionalSaccades.@preamble

T = [1 cosd(60) cosd(120) # Maps R² to H
     0 sind(60) sind(120)]'




θ = 0:0.01:(2π + 0.01)
begin
_x = sincos.(θ) .|> collect .|> reverse
_x = [x.*p./5 for (x, p) in zip(_x, θ)]
yp = last.(_x)
xp = first.(_x)
plot(xp, yp)
end
begin
_x̂ = [T] .* _x
x̂ = first.(_x̂)
ŷ = getindex.(_x̂, 2)
ẑ = getindex.(_x̂, 3)
plot(x̂, ŷ, ẑ)
end
begin
    f = Figure()
    ax = Axis(f[1, 1], aspect=1)
    xs = -1:0.01:1
    ys = -1:0.01:1
    X = Iterators.product(xs, ys)
    x = first.(X)[:]
    y = last.(X)[:]
    H = T * [x'; y']
    N = maximum(abs.(H), dims=1)[:]
    scatter!(ax, x, y, color = N.>0.8)
    lines!(ax, xp, yp)

    xi = deepcopy(_x)
    for i in eachindex(xi)
        H = T * xi[i]
        N = maximum(abs.(H))
        if N > 0.8
            xi[i+1:end] .= xi[i+1:end] .- [xi[i]]
            xi[i] = -xi[i]
            xi[i+1:end] .= xi[i+1:end] .+ [xi[i]]
        end
    end
    scatter!(ax, Point2f.(xi), color=eachindex(xi))
    f
end







begin
    # Generate 2d coordinates over a sphere
    θ = 0:0.2:(2π + 0.01)
    φ = 0:0.2:(π + 0.01)
    x = [cos(θ) .* sin(φ) for θ in θ, φ in φ][:]
    y = [sin(θ) .* sin(φ) for θ in θ, φ in φ][:]
    z = [cos(φ) for θ in θ, φ in φ][:]
    plot(x, y, z)
end
begin
    X̂ = vcat(x', y', z')
    X = inv(T[1:2, 1:2])*X̂[1:2, :]
    plot(eachrow(X)...)
end
function wrap_hex(x; R=1) # Radius of hex (Distance to nearest vertex)
    if any(x .> R)

end
