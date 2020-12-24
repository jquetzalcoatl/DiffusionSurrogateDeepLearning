using OrdinaryDiffEq, CUDA, DiffEqFlux, LinearAlgebra, Flux, DiffEqSensitivity
using Images, Plots, ArgParse
using DifferentialEquations, DelimitedFiles
using Parameters: @with_kw

function parseCommandLine()

        # initialize the settings (the description is for the help screen)
        s = ArgParseSettings(description = "Example 1 for argparse.jl: minimal usage.")

        @add_arg_table! s begin
           # "--opt1"               # an option (will take an argument)
           # "--opt2", "-o"         # another option, with short form
            "arg1"                 # a positional argument
                arg_type=Int
                default = 1
            "arg2"
                arg_type=Int
                default = 20
            "arg3"
                arg_type=String
                default = "TwoSourcesRdm"
            "arg4"
                arg_type=Int64
                default = 1
            "arg5"
              arg_type=String
              default = "/nobackup/jtoledom/DiffSolver/"
            # "arg6"
                # arg_type=Int=
        end

        return parse_args(s) # the result is a Dict{String,Any}
end

function gen_op(s)
    N=s
    Mx = Tridiagonal([1.0 for i in 1:N-1],[-2.0 for i in 1:N],[1.0 for i in 1:N-1])
    My = copy(Mx)
    # Mx[2,1] = 2.0
    # Mx[end-1,end] = 2.0
    # My[1,2] = 2.0
    # My[end,end-1] = 2.0
    return CuArray(Float32.(Mx)), CuArray(Float32.(My))
end

function f(u,p,t)
  α₁ = p
  A = u
  gMyA =  gMy * A
  gAMx = A * gMx
  gDA =  @. Dd*(gMyA + gAMx)
  # dA = @. gDA + α1*BPatch - r1*A*BPatch
  dA = @. gDA - r1*A + α₁
  return dA
end

function min_dist_to_border(source, l)
    minimum([source[1], abs(source[1]-l), source[2], abs(source[2]-l)])
end

in_source(x,xs,y,ys,r) = sqrt((x-xs)^2 + (y-ys)^2) <= r

function genInitialCond(s, radius; ns=1)
  ar = [[i,j] for i in 1:s, j in 1:s]
  dist = zeros(ns)
  init_cond = zeros(s,s)
  for ns0 in 1:ns
    source = [rand(radius:s-radius), rand(radius:s-radius)]
    dist[ns0] = min_dist_to_border(source, s)
    init_cond = init_cond .* rand() .+ [in_source(ar[i,j][1], source[1], ar[i,j][2], source[2], radius) for i in 1:s, j in 1:s]
  end
  # CuArray{Float32,2}([in_source(ar[i,j][1], source[1], ar[i,j][2], source[2], radius) for i in 1:s, j in 1:s]), dist
  CuArray{Float32,2}(init_cond/maximum(init_cond)), dist
end

function genTuples(idx, in_cond, dist; s=100, radius=5, tmax=3000.0,
            dir = "newdata")
  α₁ = in_cond

  u0 = α₁
  prob = ODEProblem{false}(f,u0,(0.0,tmax),α₁)
  @time sol = solve(prob,ROCK2(),progress=true,save_everystep=true,save_start=true)
  out = Array(sol[end])/maximum(sol[end])    #radius^2

  path = PATH * dir
  isdir(path) || mkdir(path)
  writedlm(path * "/Cell_$idx.dat", reshape(Array(α₁),:))
  writedlm(path * "/Field_$idx.dat", reshape(out,:))
  writedlm(path * "/Dist_$idx.dat", dist)
  Array(α₁), out
end


α₁, _ = genInitialCond(100, 5)
gMx, gMy = gen_op(size(α₁,1))        # This generates the discrete laplacian as in CR's tutorial
r1 = 1.0/400
Dd = 1.0
gMyA = CuArray(Float32.(zeros(size(gMx))))
gAMx = CuArray(Float32.(zeros(size(gMx))))
gDA = CuArray(Float32.(zeros(size(gMx))))
u0 = α₁

# for i in 2840:5000
#   @info i
#   in_cond = genInitialCond(100, 5)
#   init_cond, final_state = genTuples(i, in_cond)
#   f1 = heatmap(init_cond)
#   f2 = heatmap(final_state)
#   fig = plot(f1,f2, layout = @layout grid(2,1))
#   display(fig)
# end

function runFunction(start, stop, dir; ns=1)
  for i in start:stop
    @info i
    in_cond, dist = genInitialCond(100, 5; ns=ns)
    init_cond, final_state = genTuples(i, in_cond, dist; dir=dir)
    # f1 = heatmap(init_cond)
    # f2 = heatmap(final_state)
    # fig = plot(f1,f2, layout = @layout grid(2,1))
    # display(fig)
  end
end

parsed_args = parseCommandLine()
st = parsed_args["arg1"]
stop = parsed_args["arg2"]
dir_name = parsed_args["arg3"]
num_sources = parsed_args["arg4"]
PATH = parsed_args["arg5"]
runFunction(st, stop, dir_name; ns=num_sources)



############################
# α₁, _ = genInitialCond(100, 5; ns=500)
# heatmap(Array(α₁))
# gMx, gMy = gen_op(size(α₁,1))        # This generates the discrete laplacian as in CR's tutorial
# r1 = 1.0/400
# Dd = 1.0
#
# gMyA = CuArray(Float32.(zeros(size(gMx))))
# gAMx = CuArray(Float32.(zeros(size(gMx))))
# gDA = CuArray(Float32.(zeros(size(gMx))))
#
# CUDA.allowscalar(false)
#
# u0 = α₁
# prob = ODEProblem{false}(f,u0,(0.0,2000.0), α₁)
#
# @time sol = solve(prob,ROCK2(),progress=true,save_everystep=true,save_start=true)
# size(sol.t,1)#[end]
# heatmap(Array(sol[end])/maximum(sol[end]))
#
# plot([sol.t[i] for i in 1:size(sol.t,1)], [sum(Array(sol[i] .* (u0 .> 0.0))) for i in 1:size(sol.t,1)]/sum(u0 .> 0.0))
#
# sum(Array(sol[end] .*(u0 .> 0.0)))/sum(u0 .> 0.0)
#
# sum(u0 .> 0.0)
#
# maximum(sol[end])
# function stat_sol(x,y; d=1.0, γ=1.0/400, l=10.0, x0=5.0, y0=5.0, n0=10000, m0=10000)
#   sum([(2/π)^2 * sin(n*π*x0/l) * sin(m*π*y0/l) * sin(n*π*x/l) *
#     sin(m*π*y/l) * 1/d * 1/(n^2+m^2 + γ/d * l^2/pi^2) for n in 1:n0, m in 1:m0])
# end
