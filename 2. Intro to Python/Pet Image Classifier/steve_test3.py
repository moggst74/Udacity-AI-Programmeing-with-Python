# Creates empty dictionary named results_dic
results_dic = dict()

# Determines number of items in dictionary
items_in_dic = len(results_dic)
print("\nEmpty Dictionary results_dic - n items=", items_in_dic)

# Adds new key-value pairs to dictionary ONLY when key doesn't already exist. This dictionary's value is
# a List that contains only one item - the pet image label
filenames = ["beagle_0239.jpg", "Boston_terrier_02259.jpg"]
pet_labels = ["beagle", "boston terrier"]
for idx in range(0, len(filenames), 1):
    if filenames[idx] not in results_dic:
         results_dic[filenames[idx]] = [pet_labels[idx]]
    else:
         print("** Warning: Key=", filenames[idx], 
               "already exists in results_dic with value =", 
               results_dic[filenames[idx]])
