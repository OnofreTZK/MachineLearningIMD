import matplotlib.pyplot as plt

files_dir = 'mlpresults'

files = [f'./{files_dir}/number_neurons_output.dat',
         f'./{files_dir}/iterations_output.dat',
         f'./{files_dir}/learning_rate_output.dat']

for file in files:

    acc_axis = []
    par_axis = []

    try:
        results_file = open(file, 'r')

    except IOError:
        print('Check file path or file name at mlpresults!')

    result = results_file.readlines()

    # Processing
    parameter = ''

    processing_list = []
    processing_dict = {}
    parameter_mean = []

    for res in result:

        line = res.split()

        parameter = line[1]
        value = line[2]
        accuracy = float(line[4])

        parameter_mean.append(accuracy)

        if len(parameter_mean) == 4:
            mean = sum(parameter_mean)/len(parameter_mean)
            processing_dict.update({f'{mean}' : value})
            processing_list.append(mean)

            parameter_mean.clear()
    #}

    processing_list.sort()

    for acc in processing_list:
        acc_axis.append(f'{acc:.3f}')
        par_axis.append(processing_dict[f'{acc}'])

    plt.plot(par_axis, acc_axis)
    plt.ylabel('Accuracy')
    plt.xlabel(f'{parameter}')
    plt.tight_layout()
    plt.savefig(f'./{files_dir}/mlp_{parameter}.png')
    plt.clf()



