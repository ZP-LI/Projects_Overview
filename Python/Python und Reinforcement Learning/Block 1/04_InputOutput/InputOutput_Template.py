def python_wiki_text():
    ### Enter your code here ###
    text = []
    num_lines = 0
    num_words = 0
    num_characters = 0
    with open("PythonWikipedia_reversed.txt", 'r') as i_file:
        for line in i_file.readlines():
            # print(line)
            # print(len(line))
            num_characters += len(line)
            num_lines += 1
            num_words += len(line.split())
            # print(len(line.split()))
            text.append(line.split()[::-1])

    with open("PythonWikipedia.txt", 'w') as o_file:
        for line in text:
            # for i in range(len(line) - 1):
            #     line_ += (line[i] + ' ')
            # line_ += line[-1]
            line_ = ' '.join(line)

            o_file.write('{0}\n'.format(line_))

    print("Number of lines: ", num_lines)
    print("Number of words: ", num_words)
    print("Number of characters: ", num_characters)
    ### End of your code ###


if __name__ == "__main__":
    python_wiki_text()
