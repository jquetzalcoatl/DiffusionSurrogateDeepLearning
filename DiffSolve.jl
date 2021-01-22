using Plots, Flux, Statistics, LinearAlgebra, ArgParse
using Parameters: @with_kw
using CUDA
using DelimitedFiles
using Images, Random, Dates
using BSON: @save, @load

CUDA.allowscalar(false)

@with_kw struct HyperParams
    batch_size::Int=25
    NewState_input::Int=3
    Drop1::Float32=0.2
    Drop2::Float32=0.2
    Drop3::Float32=0.2
    Drop4::Float32=0.4
    weight0::Float32=15
    p1::Float32=0.05
    p2::Float32=0.95
end

function parseCommandLine(; dir=0)

        # initialize the settings (the description is for the help screen)
        s = ArgParseSettings(description = "Example 1 for argparse.jl: minimal usage.")

        @add_arg_table! s begin
           "--opt"               # an option (will take an argument)
                help = "Optmizer 1 = Momentum, 0 = ADAM, 2 = OADAM"
                arg_type = Int
                default = 0
           "--weight", "-w"         # another option, with short form
                help = "weight"
                arg_type = Float32
                default = 50f0
           "--n"
                    help = "dir name"
                    default = dir
           "--e"
                help = "epochs"
                arg_type = Int64
                default = 1000
           "--snap"
                help = "snapshots"
                arg_type = Int64
                default = 500
            "arg1"                 # a positional argument
                arg_type=Float32
                default = 0.2f0
            "arg2"
                arg_type=Float32
                default = 0.2f0
            "arg3"
                arg_type=Float32
                default = 0.2f0
            "arg4"
                arg_type=Float32
                default = 0.4f0
            "arg5"
                arg_type=Float32
                default = 20f0
            # "arg6"
                # arg_type=Int=
	    	"--p1"
                     help = "p1"
		     arg_type = Float32
                     default = 1.0f0
            "--p2"
                     help = "p2"
		     arg_type = Float32
                     default = 1.0f0
		 "--loss"               # an option (will take an argument)
 			 help = "Loss 1 = Fourth Power, 0 = Squared, 2 = MAE"
 			 arg_type = Int
 			 default = 0
		"--seed"               # an option (will take an argument)
			help = "seed Int"
			arg_type = Int
			default = 0
		"--continue", "-c"
			help = "Continue training?"
			arg_type = Int
			default = 0
		"--rollback", "-r"
			help = "use rollback"
			arg_type = Bool
			default = true
	   end

        return parse_args(s) # the result is a Dict{String,Any}
end

function save_model(model, opt ; filename = "CNN.bson", back_up=false)
	if back_up
	    isdir(PATH_out * "Models") || mkdir(PATH_out * "Models")
	    path = PATH_out * "Models/" * filename
	    isdir(PATH_out * "Models/$(parsed_args["n"])") || mkdir(PATH_out * "Models/$(parsed_args["n"])")
	    path = PATH_out * "Models/$(parsed_args["n"])/" * filename
	    NN_cpu = CNN(cpu(model.NN1), cpu(model.NN2))
	    push!(dict[:Backup], path)
	    @save string(dict[:Backup][end]) NN_cpu # opt
		# @save string(dict[:Backup][end]) model # opt
	else
		isdir(PATH_out * "Models") || mkdir(PATH_out * "Models")
	    path = PATH_out * "Models/" * filename
	    isdir(PATH_out * "Models/$(parsed_args["n"])") || mkdir(PATH_out * "Models/$(parsed_args["n"])")
	    path = PATH_out * "Models/$(parsed_args["n"])/" * filename
	    NN_cpu = CNN(cpu(model.NN1), cpu(model.NN2))
	    push!(dict[:Models], path)
	    @save string(dict[:Models][end]) NN_cpu # opt
	end
end

function load_model( ; hparams=hparams, filename = "CNN.bson", path_from_dict = "No path", back_up=false)
	if back_up
	    path_from_dict == "No path" ? (path = dict[:Models][end]; @info "Loading Model $path") :
	        (path = path_from_dict; @info "Loading Model $path")
	    model = gen_NN(hparams)
	    NN_cpu = CNN(cpu(model.NN1), cpu(model.NN2))

	    @load string(path) NN_cpu
	    # model = CNN(gpu(NN_cpu.NN1), gpu(NN_cpu.NN2), gpu(NN_cpu.NN3))
		model = CNN(gpu(NN_cpu.NN1), gpu(NN_cpu.NN2))
	    return model#, opt
	else
		path_from_dict == "No path" ? path = dict[:Models][end] :
	        (path = path_from_dict; @info "Loading Model $path")
	    model = gen_NN(hparams)
	    NN_cpu = CNN(cpu(model.NN1), cpu(model.NN2))
	    # opt = (parsed_args["opt"] == 1 ?  Momentum() : ADAM(0.0001))
	    @load string(path) NN_cpu# opt
	    model = CNN(gpu(NN_cpu.NN1), gpu(NN_cpu.NN2))
	    return model#, opt
	end
end

function load_model_CPU( ; hparams=hparams, filename = "CNN.bson", path_from_dict = "No path", back_up=false)
	path_from_dict == "No path" ? path = dict[:Models][end] :
        (path = path_from_dict; @info "Loading Model $path")
    model = gen_NN(hparams)
    NN_cpu = CNN(cpu(model.NN1), cpu(model.NN2))
    # opt = (parsed_args["opt"] == 1 ?  Momentum() : ADAM(0.0001))
    @load string(path) NN_cpu# opt
    # model = CNN(gpu(NN_cpu.NN1), gpu(NN_cpu.NN2))
    # return model#, opt
	return NN_cpu
end

function load_data(path = PATH_in)
    files = readdir(path)
    in_size = sum([split(files[i], "_")[1] for i in 1:size(files,1)] .== "Cell")
    out_size = sum([split(files[i], "_")[1] for i in 1:size(files,1)] .== "Field")
    lab_size = sum([split(files[i], "_")[1] for i in 1:size(files,1)] .== "Dist")
    in_size == out_size == lab_size ? (@info "same num of files") : (@info "more input or output or label files!")
    input = zeros(100^2, in_size)
    output = zeros(100^2, out_size)
    labs = zeros(100^2, lab_size)

    in_idx = 1
    out_idx = 1
    lab_idx = 1
    for i in 1:size(files,1)
        if split(files[i],"_")[1] == "Cell"
            input[:,in_idx] = readdlm(path * files[i]);
            in_idx +=1
        elseif split(files[i],"_")[1] == "Field"
            output[:,out_idx] = readdlm(path * files[i]);
            out_idx +=1
        else
            labs[:, lab_idx] .= readdlm(path * files[i])[1];
            lab_idx +=1
        end
    end
    # (input .> 1.0) .* (-1.0) .+ (input .< 2.0) .* input, output
    input, output, labs
end

function init_dict(args)
    dict_path = PATH_out * "Dicts"
    isdir(dict_path) || mkdir(dict_path)
    dict_path = PATH_out * "Dicts/$(parsed_args["n"])"
    isdir(dict_path) || mkdir(dict_path)
    dict = Dict()
    dict[:Path] = dict_path * "/Dict_$(now()).bson"
    dict[:Opt] = (parsed_args["opt"] == 1 ? "Momentum" : "ADAM")
	dict[:Loss] = (parsed_args["loss"] == 1 ? "Fourth" : (parsed_args["loss"] == 0 ? "Squared" : "MAE"))
    dict[:Dropout_Params] = [args.Drop1, args.Drop2, args.Drop3, args.Drop4]
	dict[:Weight] = args.weight0
	dict[:P] = [args.p1, args.p2]
	dict[:Seed] = parsed_args["seed"]
	dict[:Rollback] = parsed_args["rollback"]
    dict[:Plots] = Array{String,1}([])
    dict[:Plots2] = Array{String,1}([])
	dict[:Plots3] = Array{String,1}([])
    dict[:Models] = Array{String,1}([])
	dict[:Backup] = Array{String,1}([])
	dict[:Backup_Tr] = Array{String,1}([])
	dict[:Backup_Te] = Array{String,1}([])
	dict[:Backup_Val] = Array{String,1}([])
	dict[:Tr_idx] = Array{Int,1}([])
	dict[:Te_idx] = Array{Int,1}([])
	dict[:Va_idx] = Array{Int,1}([])

    @save dict[:Path] dict
    return dict
end

function load_dict(path)
    @load path dict
    dict
end

function gen_Rand_idx(s; tr_s = 0.7, te_s = 0.2)
    l = randperm(s)
    tr_idx = l[1:Int(round(s*tr_s))]
    te_idx = l[Int(round(s*tr_s))+1:Int(round(s*(te_s+tr_s)))]
    va_idx = l[Int(round(s*(te_s+tr_s)))+1:end]

	dict[:Tr_idx] = tr_idx
	dict[:Te_idx] = te_idx
	dict[:Va_idx] = va_idx

    @save dict[:Path] dict

    tr_idx, te_idx, va_idx
end

function genDataset( ; tr_s = 0.7, te_s = 0.2, va_s=0.1)
    input, output, labs = load_data()
    tr_idx, te_idx, va_idx = gen_Rand_idx(size(input,2))

    input1 = reshape(input[:,tr_idx],100,100,1,:)
    output1 = reshape(output[:,tr_idx],100,100,1,:) #.+ 1f-5)
    labs1 = reshape(labs[:,tr_idx],100,100,1,:)

    input_test = reshape(input[:,te_idx],100,100,1,:)
    output_test = reshape(output[:,te_idx],100,100,1,:) #.+ 1f-5)
    labs_test = reshape(labs[:,te_idx],100,100,1,:)

    input_val = reshape(input[:,va_idx],100,100,1,:)
    output_val = reshape(output[:,va_idx],100,100,1,:) #.+ 1f-5)
    labs_val = reshape(labs[:,va_idx],100,100,1,:)

    N, M, N_test, N_val = size(tr_idx, 1), hparams.batch_size, size(te_idx, 1), size(va_idx, 1)
    dataA = [(input1[:,:,:,i], output1[:,:,:,i], labs1[:,:,:,i]) for i in Iterators.partition(1:N,M)]
    dataA_test = [(input_test[:,:,:,i], output_test[:,:,:,i], labs_test[:,:,:,i]) for i in Iterators.partition(1:N_test,M)]
    dataA_val = [(input_val[:,:,:,i], output_val[:,:,:,i], labs_val[:,:,:,i]) for i in Iterators.partition(1:N_val,M)]
    dataA, dataA_test, dataA_val
end

function genDataset( dict)
    input, output, labs = load_data()
    tr_idx, te_idx, va_idx = dict[:Tr_idx], dict[:Te_idx], dict[:Va_idx]

    input1 = reshape(input[:,tr_idx],100,100,1,:)
    output1 = reshape(output[:,tr_idx],100,100,1,:) #.+ 1f-5)
    labs1 = reshape(labs[:,tr_idx],100,100,1,:)

    input_test = reshape(input[:,te_idx],100,100,1,:)
    output_test = reshape(output[:,te_idx],100,100,1,:) #.+ 1f-5)
    labs_test = reshape(labs[:,te_idx],100,100,1,:)

    input_val = reshape(input[:,va_idx],100,100,1,:)
    output_val = reshape(output[:,va_idx],100,100,1,:) #.+ 1f-5)
    labs_val = reshape(labs[:,va_idx],100,100,1,:)

    N, M, N_test, N_val = size(tr_idx, 1), hparams.batch_size, size(te_idx, 1), size(va_idx, 1)
    dataA = [(input1[:,:,:,i], output1[:,:,:,i], labs1[:,:,:,i]) for i in Iterators.partition(1:N,M)]
    dataA_test = [(input_test[:,:,:,i], output_test[:,:,:,i], labs_test[:,:,:,i]) for i in Iterators.partition(1:N_test,M)]
    dataA_val = [(input_val[:,:,:,i], output_val[:,:,:,i], labs_val[:,:,:,i]) for i in Iterators.partition(1:N_val,M)]
    dataA, dataA_test, dataA_val
end

struct CNN
    NN1
    NN2
    # NN3
end

function gen_NN(args)
    lrelu(x) = leakyrelu(x, 0.02) |> gpu
    NN1 = Chain(Conv((3,3), 1 => 4, leakyrelu; stride = 1, pad = 1),
        # x->leakyrelu.(x, 0.02),
        Dropout(args.Drop1, dims = :),
        BatchNorm(4),
        Conv((3,3), 4 => 8, leakyrelu; stride = 1, pad = 1),
        # x->leakyrelu.(x, 0.02),

        BatchNorm(8),
        Conv((3,3), 8 => 16, leakyrelu; stride = 1, pad = 1),
        # x->leakyrelu.(x, 0.02),
        BatchNorm(16),
        Conv((3,3), 16 => 8, leakyrelu; stride = 1, pad = 1),
        # x->leakyrelu.(x, 0.02),
        BatchNorm(8),
        Conv((3,3), 8 => 4, leakyrelu; stride = 1, pad = 1),
        # x->leakyrelu.(x, 0.02),
        BatchNorm(4),
        # Conv((3,3), 4 => 1, identity; stride = 1, pad = 1),
        Conv((3,3), 4 => 1, relu; stride = 1, pad = 1),
        # x->leakyrelu.(x, 0.02),
        Dropout(args.Drop2, dims = :),
        BatchNorm(1)
    ) |> gpu
    NN2 = Chain(
        Conv((3,3),1=>64, leakyrelu;stride=1,pad=1),
        BatchNorm(64),
        Dropout(args.Drop3, dims = :),
        MeanPool((2,2); pad = 0, stride = 2),
        Conv((3,3),64=>128, leakyrelu;stride=1,pad=1),

        MeanPool((2,2); pad = 1, stride = 2),
        Conv((3,3),128=>256, leakyrelu;stride=1,pad=1),

        MeanPool((2,2); pad = 0, stride = 2),
        Conv((3,3),256=>512, leakyrelu;stride=1,pad=1),

        MeanPool((2,2); pad = 0, stride = 2),
        Conv((3,3),512=>1024, leakyrelu;stride=1,pad=1),

        MeanPool((2,2); pad = 0, stride = 2),
        Conv((3,3),1024=>2048, leakyrelu;stride=1,pad=0),
        # x-> upsample(x),
        ConvTranspose((3,3),2048=>1024, leakyrelu;stride=2,pad=0),
        # # x-> upsample(x),
        ConvTranspose((3,3),1024=>512, leakyrelu;stride=2,pad=0),
        # # x-> upsample(x),
        ConvTranspose((3,3),512=>256, leakyrelu;stride=2,pad=1),
        # # # x-> upsample(x),
        ConvTranspose((3,3),256=>128, leakyrelu;stride=2,pad=1),
        # # # x-> upsample(x),
        ConvTranspose((3,3),128=>64, leakyrelu;stride=2,pad=0),
        # # # x-> upsample(x),
        Dropout(args.Drop4, dims = :),
        ConvTranspose((4,4),64=>1, relu;stride=2,pad=2),
        BatchNorm(1)
    ) |> gpu
    #NN3(x1,x2) = x1+x2
    # NN3(x1,x2) = args.p1 .* x1 +  args.p2 .* x2
	# NN3(x1,x2) = hparams.p1 .* x1 +  hparams.p2 .* x2
    #NN3 = Chain(x->reshape(x, 10000, :),
    #        LSTM(10000,10000),
    #        x->reshape(x, 100,100,1, :)) |>gpu
    return CNN(NN1, NN2)
end

NN3(x1,x2,args) = args.p1 .* x1 +  args.p2 .* x2

function gen_NN_CPU(args)
    lrelu(x) = leakyrelu(x, 0.02)
    NN1 = Chain(Conv((3,3), 1 => 4, leakyrelu; stride = 1, pad = 1),
        # x->leakyrelu.(x, 0.02),
        Dropout(args.Drop1, dims = :),
        BatchNorm(4),
        Conv((3,3), 4 => 8, leakyrelu; stride = 1, pad = 1),
        # x->leakyrelu.(x, 0.02),

        BatchNorm(8),
        Conv((3,3), 8 => 16, leakyrelu; stride = 1, pad = 1),
        # x->leakyrelu.(x, 0.02),
        BatchNorm(16),
        Conv((3,3), 16 => 8, leakyrelu; stride = 1, pad = 1),
        # x->leakyrelu.(x, 0.02),
        BatchNorm(8),
        Conv((3,3), 8 => 4, leakyrelu; stride = 1, pad = 1),
        # x->leakyrelu.(x, 0.02),
        BatchNorm(4),
        # Conv((3,3), 4 => 1, identity; stride = 1, pad = 1),
        Conv((3,3), 4 => 1, relu; stride = 1, pad = 1),
        # x->leakyrelu.(x, 0.02),
        Dropout(args.Drop2, dims = :),
        BatchNorm(1)
    )
    NN2 = Chain(
        Conv((3,3),1=>64, leakyrelu;stride=1,pad=1),
        BatchNorm(64),
        Dropout(args.Drop3, dims = :),
        MeanPool((2,2); pad = 0, stride = 2),
        Conv((3,3),64=>128, leakyrelu;stride=1,pad=1),

        MeanPool((2,2); pad = 1, stride = 2),
        Conv((3,3),128=>256, leakyrelu;stride=1,pad=1),

        MeanPool((2,2); pad = 0, stride = 2),
        Conv((3,3),256=>512, leakyrelu;stride=1,pad=1),

        MeanPool((2,2); pad = 0, stride = 2),
        Conv((3,3),512=>1024, leakyrelu;stride=1,pad=1),

        MeanPool((2,2); pad = 0, stride = 2),
        Conv((3,3),1024=>2048, leakyrelu;stride=1,pad=0),
        # x-> upsample(x),
        ConvTranspose((3,3),2048=>1024, leakyrelu;stride=2,pad=0),
        # # x-> upsample(x),
        ConvTranspose((3,3),1024=>512, leakyrelu;stride=2,pad=0),
        # # x-> upsample(x),
        ConvTranspose((3,3),512=>256, leakyrelu;stride=2,pad=1),
        # # # x-> upsample(x),
        ConvTranspose((3,3),256=>128, leakyrelu;stride=2,pad=1),
        # # # x-> upsample(x),
        ConvTranspose((3,3),128=>64, leakyrelu;stride=2,pad=0),
        # # # x-> upsample(x),
        Dropout(args.Drop4, dims = :),
        ConvTranspose((4,4),64=>1, relu;stride=2,pad=2),
        BatchNorm(1)
    )
    return CNN(NN1, NN2)
end

function (m::CNN)(x)
    x1 = m.NN1(x)
    x2 = m.NN2(x)
    # z = m.NN3(x1,x2)
	z = NN3(x1,x2, hparams)
	# z = x2
    z
end

function myPlots(dataA, gNet,l, l_test, l_val, opt; idxIn = 0, svfig = false,
        filename="plot.png", timestamp=false, modelname="CNN.bson", printplots=true, filename2="plot2.png")
    idxIn == 0 ? idx = rand(1:size(dataA,1)-1) : idx = idxIn
    x, y = dataA[idx][1], dataA[idx][2]
    testmode!(gNet, true)

    ŷ = gNet(x|>gpu) |>cpu
    im_x = reshape(x[:,:,:,1:5],100,500)
    im_ŷ = reshape(ŷ[:,:,:,1:5],100,500)
    im_y = reshape(y[:,:,:,1:5],100,500)
    for i in 1:3
        r_x = x[:,:,:,5*i + 1:5*i + 5]
        im_x = vcat(im_x, reshape(r_x, 100,500))

        r_ŷ = ŷ[:,:,:,5*i + 1:5*i + 5]
        im_ŷ = vcat(im_ŷ, reshape(r_ŷ, 100,500))

        r_y = y[:,:,:,5*i + 1:5*i + 5]
        im_y = vcat(im_y, reshape(r_y, 100,500))
    end


    fig1 = heatmap(im_x, xaxis=false, yaxis=false)
    fig2 = heatmap(im_ŷ, xaxis=false, yaxis=false)
    fig3 = heatmap(im_y, xaxis=false, yaxis=false)
    # fig4 = heatmap(abs.(im_y .- im_ŷ), xaxis=false, yaxis=false)
    # fig5 = heatmap(abs.(im_y .- im_ŷ) ./(im_y .+ 1e-5), xaxis=false, yaxis=false,
	# 		c=cgrad(:default), clims=(0,1))
	fig4 = heatmap(abs.(im_y .- im_ŷ), xaxis=false, yaxis=false,
			c=cgrad(:roma, scale = :exp))

	fig5 = heatmap(abs.(im_y .- im_ŷ) ./(im_y .+ 1e-5), xaxis=false, yaxis=false,
			c=cgrad(:roma,  [0.05, 0.1, 0.5], categorical = true), clims=(0,1),
			xticks=false)
    # fig5 = scatter(l, label="loss", xlabel=:epochs)
    fig6 = plot([Vector{Float32}(l) (l_test + l_val)/2.0],
            yscale=:log10, frame=:box, xlabel=:epochs,
            ylabel="error(MSE)", lw=2, legend=:topright, label=["Tr" "Te + V" "V"])
    fig7 = plot([Vector{Float32}(l) (l_test + l_val)/2.0],
            frame=:box, xlabel=:epochs, ylabel="error(MSE)", lw=2,
            legend=:topright, label=["Tr" "Te + V" "V"])

    y_target = dataA[idx][2]
    δy = reshape(abs.(y_target .- ŷ),:)
    # δy = measure_res(dataA, gNet)
    mv = round(mean(δy),sigdigits=3)
    mv_yt = 1.0 #mean(δy ./ reshape(h_target, :))
    h1 = plot(sort(δy, rev=true), lw=2, yscale=:log10, xlabel=:Rank,
            ylabel=:residual, label=:none, frame=:box )
    h1a = plot(sort(δy, rev=true), lw=2, xscale=:log10, yscale=:log10,
            xlabel=:Rank, ylabel=:residual, label=:none, frame=:box )
    h2 = histogram(δy, normalize=true, frame=:box, xlabel=:residual,
            ylabel="PDF", lw=2, label="<res> = $mv")
    h2a = histogram(δy ./ reshape(y_target,:), xrange=(0.1,1.0), normalize=true,
            frame=:box, xlabel=:residual, ylabel="PDF", lw=2, label="<res> = $mv_yt")
    h3 = scatter( reshape(y_target|>cpu,:), δy, frame=:box,
            xlabel="target value pixelwise", ylabel="residual", legend=:none)
    h3a = scatter( reshape(y_target|>cpu,:), δy ./ (reshape(y_target|>cpu,:) .+ 1e-5),
        yrange=(0.0,0.3), xrange=(0.0,1.0), frame=:box, xlabel="target value pixelwise", ylabel="residual/val", legend=:none)

    f = plot(fig1, fig2, fig3,
            fig6, fig4, fig5,
            fig7, h1, h1a,
            h3, h3a, h2,
        size=(2000,2000), dpi=300, left_margin = 20Plots.mm, bottom_margin = 15Plots.mm,
		layout = @layout grid(4,3))

    if svfig
        timestamp ? (dt = now(); filename = "plot_$dt.png";
        modelname = "CNN_$dt.bson"; filename2 = "plot2_$dt.png";) : nothing

        isdir(PATH_out * "Plots") || mkdir(PATH_out * "Plots")
        isdir(PATH_out * "Plots/$(parsed_args["n"])") || mkdir(PATH_out * "Plots/$(parsed_args["n"])")
        push!(dict[:Plots], PATH_out * "Plots/$(parsed_args["n"])/" * filename)
        savefig(f, dict[:Plots][end])
        save_model(gNet, opt; filename=modelname)
        @save dict[:Path] dict
    end

    printplots ? display(f) : nothing
end

function myPlots(dataA, gNet; idxIn=0, dims=[5,5], svfig = false,
        filename="plot2.png", timestamp=false, printplots=true)
    idxIn == 0 ? idx = rand(1:size(dataA,1)-1) : idx = idxIn
    δ∇y_x, δ∇y_y = get_gradient(dataA, gNet; idx=idx)

    im_x = reshape(δ∇y_x[:,:,:,1:dims[2]],99,100*dims[2])
    im_y = reshape(δ∇y_y[:,:,:,1:dims[2]],100,99*dims[2])
    for i in 1:dims[1]-1
        r_x = δ∇y_x[:,:,:,dims[2]*i + 1:dims[2]*i + dims[2]]
        im_x = vcat(im_x, reshape(r_x, 99,100*dims[2]))

        r_y = δ∇y_y[:,:,:,dims[2]*i + 1:dims[2]*i + dims[2]]
        im_y = vcat(im_y, reshape(r_y, 100,99*dims[2]))
    end

    fig1 = heatmap(im_x, xaxis=false, yaxis=false)
    fig2 = heatmap(im_y, xaxis=false, yaxis=false)

    h1x = plot(sort(reshape(δ∇y_x,:), rev=true), lw=2, yscale=:log10, xlabel=:Rank,
            ylabel=:residual, label=:none, frame=:box )
    h1y = plot(sort(reshape(δ∇y_y,:), rev=true), lw=2, yscale=:log10, xlabel=:Rank,
            ylabel=:residual, label=:none, frame=:box )

    h2x = plot(sort(reshape(δ∇y_x,:), rev=true), lw=2, xscale=:log10, yscale=:log10,
            xlabel=:Rank, ylabel=:residual, label=:none, frame=:box )
    h2y = plot(sort(reshape(δ∇y_y,:), rev=true), lw=2, xscale=:log10, yscale=:log10,
            xlabel=:Rank, ylabel=:residual, label=:none, frame=:box )
    mv_x = round(mean(δ∇y_x),sigdigits=3)
    mv_y = round(mean(δ∇y_y),sigdigits=3)
    h3x = histogram(reshape(δ∇y_x,:), normalize=true, frame=:box, xlabel=:residual,
            ylabel="PDF", lw=2, label="<res> = $mv_x")
    h3y = histogram(reshape(δ∇y_y,:), normalize=true, frame=:box, xlabel=:residual,
            ylabel="PDF", lw=2, label="<res> = $mv_y")

    f = plot(fig1, fig2, h1x, h1y, h2x, h2y, h3x, h3y, dpi=300, size=(1000,1000),
                left_margin = 20Plots.mm, bottom_margin = 15Plots.mm, layout = @layout grid(4,2))

    if svfig
        timestamp ? (dt = now(); filename = "plot2_$dt.png";) : nothing

        isdir(PATH_out * "Plots") || mkdir(PATH_out * "Plots")
        isdir(PATH_out * "Plots/$(parsed_args["n"])") || mkdir(PATH_out * "Plots/$(parsed_args["n"])")
        push!(dict[:Plots2], PATH_out * "Plots/$(parsed_args["n"])/" * filename)
        savefig(f, dict[:Plots2][end])

        @save dict[:Path] dict
    end

    printplots ? display(f) : nothing
end

function myPlots(dataA, g, c::Int, dict; sz=5, svfig = false,
	filename="plot3.png", timestamp=false, printplots=true)

    x, y = get_samp(dataA, g; sz0=sz)
    fig = scatter( x, y, frame=:box,
        xlabel="Prediction (|ŷ⟩)", ylabel="True (|y⟩)", legend=:none,
        xrange=(-0.1,1.1), yrange=(-0.1,1.1), ms=0.4, tickfont = font(13, "Helvetica"))
    # fig = marginalhist(reshape(ŷ|>cpu,:), reshape(y_target|>cpu,:), frame=:box,
    #     xlabel="output", ylabel="true", legend=:none, size=(600,600))
    # fig = corrplot([reshape(ŷ|>cpu,:) reshape(y_target|>cpu,:)], frame=:box,
    #     xlabel="output", ylabel="true", legend=:none, size=(600,600))
    fig = plot!(0:1,x->x, lw=2, c=:red)
    f2 = plot(    histogram(x, normalize=true, xrange=(-0.1,1.1)),     plot(framestyle = :none),
             fig,
             histogram(y, xrange=(0,25.1), yrange=(-0.1,1.1), orientation = :horizontal,
             normalize=true),
         link = :both, size=(800,800), left_margin = 20Plots.mm, bottom_margin = 10Plots.mm)

	f_temp = plot(x, seriestype=:barhist, width=0, normalize=true, xrange=(-0.1,1.1),
	 	left_margin = 10Plots.mm, bottom_margin = 10Plots.mm, frame=:box, label="Prediction",
		xlabel="Field", ylabel="PDF", tickfont = font(13, "Helvetica"), opacity=0.7)
	f_temp = plot!(y, seriestype=:barhist, width=0, normalize=true, xrange=(-0.1,1.1),
	 	left_margin = 10Plots.mm, bottom_margin = 10Plots.mm, frame=:box, label="True",
		xlabel="Field", ylabel="PDF", tickfont = font(13, "Helvetica"), opacity=0.7)
    # svifig ? savefig(f2, path) : display(f2)

	if svfig
        timestamp ? (dt = now(); filename = "plot3_$c_$dt.png";) : nothing

        isdir(PATH_out * "Misc") || mkdir(PATH_out * "Misc")
		isdir(PATH_out * "Misc/Pl") || mkdir(PATH_out * "Misc/Pl")
        # isdir(PATH_out * "Plots/$(parsed_args["n"])") || mkdir(PATH_out * "Plots/$(parsed_args["n"])")
        push!(dict[:Plots3], PATH_out * "Misc/Pl/" * filename)
        savefig(f2, dict[:Plots3][end])
		###
		push!(dict[:Plots3], PATH_out * "Misc/Pl/Hist_" * filename)
		f, s1 = get_slices(x,y; nslices=20)
		fig2 = plot(f..., size=(2800,1800),
				left_margin = 5Plots.mm, top_margin = 10Plots.mm,
				bottom_margin = 10Plots.mm, right_margin = 10Plots.mm,
				tickfont = font(13, "Helvetica"))
        savefig(fig2, dict[:Plots3][end])

		data_path = split(PATH_out * "Misc/Pl/Data_" * filename,".")[1] * ".dat"
		writedlm(data_path, s1)

		savefig(f_temp, PATH_out * "Misc/Pl/Histogram_" * filename)
		###

        @save dict[:Path] dict
    end

    printplots ? display(f2) : nothing
end

function myPlots(dataA, g, c::Int, dict, cc::Int; sz=5, svfig = false,
	filename="plot4.png", timestamp=false, printplots=true)

    x, y = get_samp(dataA, g; sz0=sz)
	f, _ = get_slices(x,y; nslices=20)
	fig = plot(f..., size=(2800,1800),
			left_margin = 5Plots.mm, top_margin = 10Plots.mm,
			bottom_margin = 10Plots.mm, right_margin = 10Plots.mm,
			tickfont = font(13, "Helvetica"))

	if svfig
        timestamp ? (dt = now(); filename = "plot4_$c_$dt.png";) : nothing

        isdir(PATH_out * "Misc") || mkdir(PATH_out * "Misc")
		isdir(PATH_out * "Misc/Pl") || mkdir(PATH_out * "Misc/Pl")
        # isdir(PATH_out * "Plots/$(parsed_args["n"])") || mkdir(PATH_out * "Plots/$(parsed_args["n"])")
        push!(dict[:Plots3], PATH_out * "Misc/Pl/" * filename)
        savefig(fig, dict[:Plots3][end])

        @save dict[:Path] dict
    end

    printplots ? display(fig) : nothing
end

function get_samp(dataA, g; sz0=5)
    nperb = prod(size(dataA[1][1]))
    sz = minimum([sz0, size(dataA,1)])
    y1 = zeros(sz*nperb)
    y2 = zeros(sz*nperb)
    for idx in 1:sz
        x, y = dataA[idx][1], dataA[idx][2]
		ŷ = g(x|>gpu) |>cpu
        #ŷ = g.NN1(x|>gpu) |>cpu
        y_target = dataA[idx][2]
        y1[1 + (idx-1)*nperb:idx*nperb] = reshape(ŷ,:)
        y2[1 + (idx-1)*nperb:idx*nperb] = reshape(y_target,:)
    end
    y1,y2
end

function get_slices(xx,yy; nslices = 10)
        xθ, yθ = 1/√2*(xx .- yy), 1/√2*(xx .+ yy)
        # f1 = histogram(yθ, label="y rotated", frame=:box, normalized=true,
        #         title="total points = $(sum(size(yθ,1)))")
		f1 = plot(yθ, seriestype=:barhist, width=0, label="y=x", frame=:box, normalized=true,
                title="total points = $(sum(size(yθ,1)))")
        f = [f1]
		Δs = 1/nslices
		stats = zeros(nslices,2)
        for i in 1:nslices
                up = floor(Δs*i, sigdigits=2)
                down = floor(Δs*(i-1), sigdigits=2)
                yl = findall(x->down<x<up, yθ)
				stats[i, 1], stats[i, 2] = mean(xθ[yl]), std(xθ[yl])
                # h = histogram(xθ[yl],
                #         title="$down<y rot<$up, n=$(round(sum(size(yl,1))/sum(size(yθ,1)),sigdigits=3))",
                #         label="mean=$(round(mean(xθ[yl]),sigdigits=3)) \n std=$(round(std(xθ[yl]),sigdigits=3))",
                #         frame=:box, normalized=true)
				# h = plot(xθ[yl], seriestype=:barhist, width=0,
                #         title="$down<field value<$up, density=$(round(sum(size(yl,1))/sum(size(yθ,1)),sigdigits=2))",
                #         label="m=$(round(mean(xθ[yl]),sigdigits=2)) \n s=$(round(std(xθ[yl]),sigdigits=2))",
                #         frame=:box, normalized=true)
				h = plot(xθ[yl], seriestype=:barhist, width=0,
                        title="$down<field value<$up", legend=false,
                        label="m=$(round(mean(xθ[yl]),sigdigits=2)) \n s=$(round(std(xθ[yl]),sigdigits=2))",
                        frame=:box, normalized=true, xlabel="|ŷ⟩ - |y⟩", ylabel="PDF",
						xlim=(stats[i, 1] - 4*stats[i, 2],stats[i, 1] + 4*stats[i, 2]),
						xticks=([-floor(4*stats[i, 2], sigdigits=1), 0, floor(4*stats[i, 2], sigdigits=1)]))

				m, s = stats[i, 1], stats[i, 2]
				lowlim, uplim = m - 4 * s, m + 4 * s
				Δx = (uplim - lowlim) * 0.01
				h = plot!(lowlim:Δx:uplim, x-> 1/sqrt(2*π*s^2)*exp(-(x-m)^2/(2*s^2)), label="N(m,s)", c=:red)
                push!(f,h)
        end
        f, stats
end

function plot_truevsoutput(dataA, g; idxIn=0, svifig = false, path = "./Plot_output_vs_true.png")
        idxIn == 0 ? idx = rand(1:size(dataA,1)-1) : idx = idxIn
        x, y = dataA[idx][1], dataA[idx][2]
        ŷ = g(x|>gpu) |>cpu
		#ŷ = g.NN1(x|>gpu) |>cpu
        y_target = dataA[idx][2]
        fig = scatter( reshape(ŷ|>cpu,:), reshape(y_target|>cpu,:), frame=:box,
            xlabel="Prediction", ylabel="True", legend=:none, size=(600,600))
        fig = plot!(0:1,x->x, lw=2, c=:red)
        svifig ? savefig(fig, path) : display(fig)
end

function get_gradient(dataA, gNet; idx=1)
    x = dataA[idx][1] |>gpu
    y = dataA[idx][2]
    ŷ = gNet(x) |>cpu

    δ∇y_x = abs.((ŷ[1:end-1,:,:,:] .- ŷ[2:end,:,:,:]) .- (y[1:end-1,:,:,:] .- y[2:end,:,:,:]))
    δ∇y_y = abs.((ŷ[:,1:end-1,:,:] .- ŷ[:,2:end,:,:]) .- (y[:,1:end-1,:,:] .- y[:,2:end,:,:]))

    return δ∇y_x, δ∇y_y
end

function measure_res(dataA, gNet)
    δy = zeros(size(dataA,1),prod(size(dataA[1][1])))
    idx=1
    for (x, y, dist) in dataA
        x = x |>gpu
        y = y |>gpu
        ŷ = gNet(x)
        δy[idx,:] = reshape(abs.(y .- ŷ),:) |> cpu
        idx +=1
    end
    δy
end

function measure_res(dataA, gNet, a)
    @info "Measure Grad res"
    δ∇y_x = zeros(size(dataA,1),25*100*99)
    δ∇y_y = zeros(size(dataA,1),25*100*99)
    rs(x) = reshape(x,:)

    for idx in 1:size(dataA,1)
        δ∇y_x[idx,:], δ∇y_y[idx,:] = rs.(get_gradient(dataA, gNet; idx=idx))
    end
    δ∇y_x, δ∇y_y
end

function custom_train!(dataA, ps, opt, gNet, loss)
    loss_list=[]
    testmode!(gNet, false)
	l_rand = Random.randperm(size(dataA,1))
    for (x, y, dist) in dataA[l_rand]
        x = x |>gpu
        y = y|>gpu
        dist = dist |> gpu
        #gs = gradient(() -> loss(x,y, dist), ps)
        gs = gradient(() -> loss(x,y), ps)
        Flux.update!(opt, ps, gs)
        #append!(loss_list, loss(x,y, dist))
        append!(loss_list, loss(x,y))
    end
    mean(loss_list)
end

function measure_loss(dataA_test, dataA_val, gNet, loss)
    loss_list_test, loss_list_val=[], []
    testmode!(gNet, true)
    for (x, y, dist) in dataA_test
        x = x |>gpu
        y = y |>gpu
        dist = dist |> gpu
        #append!(loss_list_test, loss(x,y,dist))
        append!(loss_list_test, loss(x,y))
    end
    for (x, y, dist) in dataA_val
        x = x |>gpu
        y = y |>gpu
        dist = dist |> gpu
        #append!(loss_list_val, loss(x,y,dist))
        append!(loss_list_val, loss(x,y))
    end
    mean(loss_list_test), mean(loss_list_val)
end

function backup(gNet, flag; modelname="CNN_BKP.bson", mean_loss=[0], mean_loss_test=[0], mean_loss_val=[0])
	if flag == 1
		save_model(gNet, 0; filename=modelname, back_up=true)
		save_loss_data(mean_loss, mean_loss_test, mean_loss_val)
	elseif flag == 0
		gNet = load_model(path_from_dict = dict[:Backup][end])
		return gNet
	end
end

function save_loss_data(ml, ml_te, ml_val)
	isdir(PATH_out * "Data") || mkdir(PATH_out * "Data")
	isdir(PATH_out * "Data/$(parsed_args["n"])") || mkdir(PATH_out * "Data/$(parsed_args["n"])")
	push!(dict[:Backup_Tr], PATH_out * "Data/$(parsed_args["n"])/" * "loss_Tr.dat")
	writedlm(dict[:Backup_Tr][end], ml)

	push!(dict[:Backup_Te], PATH_out * "Data/$(parsed_args["n"])/" * "loss_Te.dat")
	writedlm(dict[:Backup_Te][end], ml_te)

	push!(dict[:Backup_Val], PATH_out * "Data/$(parsed_args["n"])/" * "loss_Val.dat")
	writedlm(dict[:Backup_Val][end], ml_val)
end

function train( ; epochs=200, snap=25)
	e_len, e_thrs, rb = 22, 10, 0
	Random.seed!(dict[:Seed])
    dataA, dataA_test, dataA_val = genDataset()
    gNet = gen_NN(hparams)
    # loss(x,y; gNet=gNet) = ( ŷ1 = gNet(x) ; Flux.mse(ŷ1,y))
    # loss(x,y, w; gNet=gNet) = ( ŷ1 = gNet(x); expw = exp.(-w/hparams.weight0) ; mean( expw .* (ŷ1 .- y) .^2))
    # loss(x,y; gNet=gNet) = (ŷ1 = gNet(x); mean(abs2, exp.(-(1 .- y) ./(2 * hparams.weight0)) .* (ŷ1 .- y)))
    #loss(x,y; gNet=gNet) = (ŷ1 = gNet(x); mean((exp.(-(1 .- y) ./(4 * hparams.weight0)) .* (ŷ1 .- y)) .^ 4))
    # loss(x,y; gNet=gNet) = (ŷ1 = gNet(x); mean((exp.(-(1 .- y) ./(6 * hparams.weight0)) .* (ŷ1 .- y)) .^ 6))

	loss(x,y; gNet=gNet) = parsed_args["loss"] == 0 ?  (ŷ1 = gNet(x);
		mean(abs2, exp.(-(1 .- y) ./(2 * hparams.weight0)) .* (ŷ1 .- y))) : (parsed_args["loss"] == 1 ?
        (ŷ1 = gNet(x); mean((exp.(-(1 .- y) ./(4 * hparams.weight0)) .* (ŷ1 .- y)) .^ 4)) : (ŷ1 = gNet(x);
		mean(abs, exp.(-(1 .- y) ./( hparams.weight0)) .* (ŷ1 .- y))))

    ps = params(gNet.NN1, gNet.NN2)
    # opt = ADAM(0.0001)
    # opt = (parsed_args["opt"] == 1 ?  Momentum() : ADAM(0.00001))
    opt = parsed_args["opt"] == 1 ?  Momentum() : (parsed_args["opt"] == 0 ?
        ADAM(0.000005) : Flux.OADAM(0.0001))

    mean_loss, mean_loss_test, mean_loss_val = [], [], []
    myPlots(dataA_test, gNet,[1.0], [1.0], [1.0], opt; printplots=false)
    for epoch in 1:epochs
        @info epoch
        l = custom_train!(dataA, ps, opt, gNet, loss)
        @info l
        l_test, l_val = measure_loss(dataA_test, dataA_val, gNet, loss)
        append!(mean_loss, l)
        append!(mean_loss_test, l_test)
        append!(mean_loss_val, l_val)
        epoch % snap == 0 ? (myPlots(dataA_test, gNet, mean_loss,
                mean_loss_test, mean_loss_val, opt; svfig=true, timestamp=true,
                printplots=false);
				myPlots(dataA_test, gNet; svfig=true, timestamp=true,
                        printplots=false)) : nothing
		if epoch > 0.9 * epochs && minimum(mean_loss_test[end-20:end]) == mean_loss_test[end]
			myPlots(dataA_test, gNet, mean_loss,
	                mean_loss_test, mean_loss_val, opt; svfig=true, timestamp=true,
	                printplots=false);
			myPlots(dataA_test, gNet; svfig=true, timestamp=true,
                    printplots=false);
			break;
		end

		if epoch >= e_len
			δ_av = mean(abs.(mean_loss[end-e_len+2:end-1] - mean_loss[end-e_len+1:end-2]))
			δ = mean_loss[end] - mean_loss[end-1]
			if δ > e_thrs* δ_av && dict[:Rollback]
				pop!(mean_loss)
				pop!(mean_loss_test)
				pop!(mean_loss_val)
				gNet = backup(gNet, 0)
				ps = params(gNet.NN1, gNet.NN2)
				rb = rb + 1
			end
			@info "Average delta: $δ_av "
			@info "delta: $δ "
		end
		epoch % 5 == 0 ? backup(gNet, 1; mean_loss=mean_loss, mean_loss_test=mean_loss_test, mean_loss_val=mean_loss_val) : nothing
    end
	dict[:N_Rollbacks] = rb
	dict[:Final_state] = "Training ended succesfully!"
	@save dict[:Path] dict
    mean_loss, mean_loss_test, mean_loss_val, gNet, opt
end

function continue_train(; epochs=200, snap=25)
	e_len, e_thrs, rb = 22, 10, 0
	# Random.seed!(dict[:Seed])
    dataA, dataA_test, dataA_val = genDataset(dict)
	gNet = gen_NN(hparams)
	try
    	gNet = load_model(path_from_dict = dict[:Backup][end])
	catch e
		@warn e
		@warn "Loading Saved model"
		gNet = load_model(path_from_dict = dict[:Models][end])
	end
    # loss(x,y; gNet=gNet) = ( ŷ1 = gNet(x) ; Flux.mse(ŷ1,y))
    # loss(x,y, w; gNet=gNet) = ( ŷ1 = gNet(x); expw = exp.(-w/hparams.weight0) ; mean( expw .* (ŷ1 .- y) .^2))
    # loss(x,y; gNet=gNet) = (ŷ1 = gNet(x); mean(abs2, exp.(-(1 .- y) ./(2 * hparams.weight0)) .* (ŷ1 .- y)))
    #loss(x,y; gNet=gNet) = (ŷ1 = gNet(x); mean((exp.(-(1 .- y) ./(4 * hparams.weight0)) .* (ŷ1 .- y)) .^ 4))
    # loss(x,y; gNet=gNet) = (ŷ1 = gNet(x); mean((exp.(-(1 .- y) ./(6 * hparams.weight0)) .* (ŷ1 .- y)) .^ 6))

	loss(x,y; gNet=gNet) = parsed_args["loss"] == 0 ?  (ŷ1 = gNet(x);
		mean(abs2, exp.(-(1 .- y) ./(2 * hparams.weight0)) .* (ŷ1 .- y))) : (parsed_args["loss"] == 1 ?
        (ŷ1 = gNet(x); mean((exp.(-(1 .- y) ./(4 * hparams.weight0)) .* (ŷ1 .- y)) .^ 4)) : (ŷ1 = gNet(x);
		mean(abs, exp.(-(1 .- y) ./( hparams.weight0)) .* (ŷ1 .- y))))

    ps = params(gNet.NN1, gNet.NN2)
    # opt = ADAM(0.0001)
    # opt = (parsed_args["opt"] == 1 ?  Momentum() : ADAM(0.00001))
    opt = parsed_args["opt"] == 1 ?  Momentum() : (parsed_args["opt"] == 0 ?
        ADAM(0.000005) : Flux.OADAM(0.0001))

    mean_loss = reshape(readdlm(dict[:Backup_Tr][end]),:)
	mean_loss_test = reshape(readdlm(dict[:Backup_Te][end]),:)
	mean_loss_val = reshape(readdlm(dict[:Backup_Val][end]),:)
    # myPlots(dataA_test, gNet,[1.0], [1.0], [1.0], opt)
    for epoch in 1:epochs
        @info epoch
        l = custom_train!(dataA, ps, opt, gNet, loss)
        @info l
        l_test, l_val = measure_loss(dataA_test, dataA_val, gNet, loss)
        append!(mean_loss, l)
        append!(mean_loss_test, l_test)
        append!(mean_loss_val, l_val)
        epoch % snap == 0 ? (myPlots(dataA_test, gNet, mean_loss,
                mean_loss_test, mean_loss_val, opt; svfig=true, timestamp=true,
                printplots=false);
				myPlots(dataA_test, gNet; svfig=true, timestamp=true,
                        printplots=false)) : nothing
		if epoch > 0.9 * epochs && minimum(mean_loss_test[end-20:end]) == mean_loss_test[end]
			myPlots(dataA_test, gNet, mean_loss,
	                mean_loss_test, mean_loss_val, opt; svfig=true, timestamp=true,
	                printplots=false);
			myPlots(dataA_test, gNet; svfig=true, timestamp=true,
                    printplots=false);
			break;
		end

		if epoch >= e_len
			δ_av = mean(abs.(mean_loss[end-e_len+2:end-1] - mean_loss[end-e_len+1:end-2]))
			δ = mean_loss[end] - mean_loss[end-1]
			if δ > e_thrs* δ_av && dict[:Rollback]
				pop!(mean_loss)
				pop!(mean_loss_test)
				pop!(mean_loss_val)
				gNet = backup(gNet, 0)
				ps = params(gNet.NN1, gNet.NN2)
				rb = rb + 1
			end
			@info "Average delta: $δ_av "
			@info "delta: $δ "
		end
		epoch % 5 == 0 ? backup(gNet, 1; mean_loss=mean_loss, mean_loss_test=mean_loss_test, mean_loss_val=mean_loss_val) : nothing
    end
	dict[:N_Rollbacks] = rb
	dict[:Final_state] = "Training ended succesfully!"
	@save dict[:Path] dict
    mean_loss, mean_loss_test, mean_loss_val, gNet, opt
end
