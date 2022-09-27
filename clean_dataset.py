from os import listdir, remove

dataset_filename_list = listdir('./dataset')

#for i in range(0, 50):
#    print(dataset_filename_list[i])

class1 = 'staffordshire_bull_terrier'
class2 = 'wheaten_terrier'
class3 = 'samoyed'
class4 = 'Bombay'
class5 = 'Ragdoll'

def filter_my_classes(name):

    return not (name.startswith(class1) or name.startswith(class2) or name.startswith(class3)
    or name.startswith(class4) or name.startswith(class5))

filtered_dataset_name_list = filter(filter_my_classes, dataset_filename_list)

for file in filtered_dataset_name_list:
    filepath = f'./dataset/{file}'
    remove(filepath)
    print(file)


