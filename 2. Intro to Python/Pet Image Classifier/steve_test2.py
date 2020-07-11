dognames_dic = dict()
#dognames_dic = {'chihuahua': 1, 'beagle': 1, 'dalmatian, coach dog, carriage dog': 1}
#dognames_dic = {'chihuahua': 1, 'beagle': 1, 'dalmatian, coach dog, carriage dog': 1,'dog': 1}
dognames_dic = {'chihuahua': 1, 'beagle': 1, 'dalmatian': 1,'dog': 1}

results_dic = dict()
#results_dic = {'Dalmatian_04037.jpg': ['dalmatian', 'coach dog', 'carriage dog', 1, 0, 0], 'Boston_terrier_02285.jpg': ['boston terrier', #'boston bull, boston terrier', 1, 0, 0], 'Dog_001.jpg':['dog', 1, 0, 0]}
#results_dic = {'Dog.jpg': ['dog', 1]}
results_dic = {'Dalmatian_04037.jpg': ['dalmatian', 'coach dog', 'carriage dog', 1, 0, 0]}

print(results_dic['Dalmatian_04037.jpg'][0] in dognames_dic)
#print('dognames_dic :')

#print(dognames_dic)
#if results_dic['Dalmatian_04037.jpg'][0] in dognames_dic:
#    print('Steve match')
#else:
#    print('Steve no match')