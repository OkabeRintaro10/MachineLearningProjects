# Python code to find frequency of each word
def freq(str):

	# break the string into list of words
	str = str.split()		
	str2 = []

	# loop till string values present in list str
	for i in str:			

		# checking for the duplicacy
		if i not in str2:

			# insert value in str2
			str2.append(i)
			
	for i in range(0, len(str2)):

		# count the frequency of each word(present
		# in str2) in str and print
		print('Frequency of', str2[i], 'is :', str.count(str2[i]))

str ='apple mango apple orange orange apple guava mango mango'
freq(str)		

print()

# Using Dictionary

def count(elements):
	# check if each word has '.' at its last. If so then ignore '.'
	if elements[-1] == '.':
		elements = elements[0:len(elements) - 1]

	# if there exists a key as "elements" then simply
	# increase its value.
	if elements in dictionary:
		dictionary[elements] += 1

	# if the dictionary does not have the key as "elements"
	# then create a key "elements" and assign its value to 1.
	else:
		dictionary.update({elements: 1})


# driver input to check the program.
str = "apple mango apple orange orange apple guava mango mango"

# Declare a dictionary
dictionary = {}

# split all the word of the string.
lst = str.split()

# take each word from lst and pass it to the method count.
for elements in lst:
	count(elements)

# print the keys and its corresponding values.
for allKeys in dictionary:
	print ("Frequency of ", allKeys, end = " ")
	print (":", end = " ")
	print (dictionary[allKeys], end = " ")
	print()
