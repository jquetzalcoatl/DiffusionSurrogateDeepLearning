include("DiffSolve.jl")


parsed_args = parseCommandLine(dir=1)

PATH_in = "/PATH/TO/GENERATED/DATA/TwoSources/"
PATH_out = "/PATH/OUT/"
hparams = HyperParams(Drop1 = parsed_args["arg1"], Drop2 = parsed_args["arg2"],
                Drop3 = parsed_args["arg3"], Drop4 = parsed_args["arg4"],
		weight0 = parsed_args["weight"], p1 = parsed_args["p1"],
		p2 = parsed_args["p2"])
# hparams = HyperParams(Drop1=0.3, Drop2=0.3, Drop3=0.3, Drop4=0.1, weight0=1)

cont = parsed_args["continue"]
dir = parsed_args["n"]
if cont == 0
        dict = init_dict(hparams)
        mean_loss, mean_loss_test, mean_loss_val, g, opt = train( ; epochs=parsed_args["e"], snap=parsed_args["snap"])
elseif cont == 1
        dict_filename = readdir(PATH_out * "Dicts/$dir/")[end]
        dict = load_dict(PATH_out * "Dicts/$dir/" * dict_filename)
        # mean_loss, mean_loss_test, mean_loss_val, g, opt = continue_train( ; epochs=parsed_args["e"], snap=parsed_args["snap"])
		mean_loss, mean_loss_test, mean_loss_val, g, opt = continue_train( ; epochs=50, snap=25)
end
