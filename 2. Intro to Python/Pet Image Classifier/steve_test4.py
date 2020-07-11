#dognames_dic = dict()
#dognames_dic = {'dalmatian':['dalmatian, coach dog, carriage dog', 1]}
#strip_line = 'coach dog'

#test_file ='this is a string'
#test2 = 'this'

#print('steve22')
#print(test2 in test_file)
#'this' in 'this is a string'

dog_file = ['chihuahua','japanese spaniel','maltese dog, maltese terrier, maltese','shih-tzu']
strip_line = 'maltese dog'   
    
if strip_line in dog_file[1]:
#    dognames_dic[strip_line] = 1
    print('duplicate!')
    print(strip_line)
else:
    print('not duplicate!')
    print(strip_line)
