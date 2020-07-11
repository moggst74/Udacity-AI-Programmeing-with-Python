#Imports only listdir function from OS module
#from os import listdir

#assign list of files in directory to a variable file_name
#file_name = listdir("pet_images/")

#pet_image = "Boston_terrier_02259.jpg"
#pet_label = ""
           
#make file_name lower case
#low_pet_image = pet_image.lower()
           
#split low_pet_image by "-"
#word_list_pet_image = low_pet_image.split("_")
           
#for word in word_list_pet_image:
#    if word.isalpha():
#        pet_label += word + " "
        
#print(pet_label)

# Defining lists to populate dictionary 
filenames = ["Beagle_01141.jpg", "Beagle_01125.jpg", "skunk_029.jpg" ]
pet_labels = ["beagle", "beagle", "skunk"]
classifier_labels = ["walker hound, walker foxhound", "beagle",
                     "skunk, polecat, wood pussy"]
pet_label_is_dog = [1, 1, 0]
classifier_label_is_dog = [1, 1, 0]

# Defining empty dictionary
results_dic = dict()

# Populates empty dictionary with both labels &indicates if they match (idx 2)
for idx in range (0, len(filenames), 1):
    # If first time key is assigned initialize the list with pet & 
    # classifier labels
    if filenames[idx] not in results_dic:
        results_dic[filenames[idx]] = [ pet_labels[idx], classifier_labels[idx] ]

    # Determine if pet_labels matches classifier_labels using in operator
    # - so if pet label is 'in' classifier label it's a match
    # ALSO since Key already exists because labels were added, append 
    # value to end of list for idx 2 
    # if pet image label was FOUND then there is a match 
    if pet_labels[idx] in classifier_labels[idx]:
        results_dic[filenames[idx]].append(1)

    # if pet image label was NOT found then there is no match
    else:
        results_dic[filenames[idx]].append(0)

# Populates dictionary with whether or not labels indicate a dog image (idx 3&4)
for idx in range (0, len(filenames), 1):
    # Key already exists, extend values to end of list for idx 3 & 4
    results_dic[filenames[idx]].extend([pet_label_is_dog[idx], 
                                       classifier_label_is_dog[idx]])

# Iterates through the list to print the results for each filename
for key in results_dic:
    print("\nFilename=", key, "\npet_image Label=", results_dic[key][0],
          "\nClassifier Label=", results_dic[key][1], "\nmatch=",
          results_dic[key][2], "\nImage is dog=", results_dic[key][3],
          "\nClassifier is dog=", results_dic[key][4])                        

    # Provides classifications of the results
    if sum(results_dic[key][2:]) == 3:
        print("*Breed Match*")
    if sum(results_dic[key][3:]) == 2:
        print("*Is-a-Dog Match*")
    if sum(results_dic[key][3:]) == 0 and results_dic[key][2] == 1:
        print("*NOT-a-Dog Match*")