# Step 1: Read the file content into a string
with open('data/words.txt', 'r') as file:
    file_content = file.read()

# Step 2: Convert the string into a list where each word is an element
# Assuming each word is separated by a newline
words_list = file_content.split('\n')

# Step 3: Save the list to a new file in a format that can be easily read back into a list
import json

# Saving the list
with open('data/words_list.json', 'w') as json_file:
    json.dump(words_list, json_file)

# Reading the list back
# with open('words_list.json', 'r') as json_file:
#     words_list_read = json.load(json_file)

# words_list_read now contains the list read from the file