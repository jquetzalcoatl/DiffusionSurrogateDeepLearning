include("DiffSolve.jl")
using StatsBase

parsed_args = parseCommandLine()

PATH_in = "/PATH/TO/GENERATED/DATA/TwoSources/"
PATH_out = "/PATH/OUT/"

hparams = HyperParams(Drop1 = parsed_args["arg1"], Drop2 = parsed_args["arg2"],
                Drop3 = parsed_args["arg3"], Drop4 = parsed_args["arg4"],
				weight0 = parsed_args["weight"], p1 = parsed_args["p1"],
				p2 = parsed_args["p2"])



#function get_stats()
l = length(readdir(PATH_out * "/Plots/"))
delta = zeros(l,10)
rs(x)=reshape(x,:)

for i in 1:l
    try
        d_path = readdir(PATH_out * "Dicts/$i/")
        dict_load = load_dict( PATH_out * "Dicts/$i/" * d_path[end])
		hparams = HyperParams(Drop1 = dict[:Dropout_Params][1], Drop2 = dict[:Dropout_Params][2],
		                Drop3 = dict[:Dropout_Params][3], Drop4 = dict[:Dropout_Params][4],
				weight0 = dict[:Weight], p1 = dict[:P][1], p2 = dict[:P][2])
		dataA, dataA_test, dataA_val = genDataset(dict_load)
        g = load_model(path_from_dict = dict_load[:Models][end])
        delta[i,1] = i
        d1 = reshape(measure_res(dataA_test, g),:)
        #delta[i,1] = i
        delta[i,2] = mean(d1)
        delta[i,3] = maximum(d1)
        delta[i,4] = percentile(d1,99)
        dx, dy = rs.(measure_res(dataA_test, g,1))

        delta[i,5] = mean(dx)
        delta[i,6] = maximum(dx)
        delta[i,7] = percentile(dx,99)

        delta[i,8] = mean(dy)
        delta[i,9] = maximum(dy)
        delta[i,10] = percentile(dy,99)
        # mean_d_grad[i] = mean(mean.(measure_res(dataA_test, g,1)))
		# plot_truevsoutput(dataA_test, g; svifig = true, path = "./Pl/Plot_output_vs_true$i.png")
		myPlots(dataA, g, 0, dict_load; sz=150, svfig=true, timestamp=false, printplots=false,
				filename="Plot_output_vs_true_tr_$i.png");
		myPlots(dataA_test, g, 1, dict_load; sz=150, svfig=true, timestamp=false, printplots=false,
				filename="Plot_output_vs_true_te_$i.png");
    catch e
		@show "Model with directory $i does not have saved Model or Dict"
    end
end
writedlm("./residuals.csv", delta)
