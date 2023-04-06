
def combine_words():
    word_list = [
        ['', 'PROCESSOR', 'CPU', 'THREAD'],
        ['', 'cache', 'fabric', 'interconnect', 'protocol'],
        [
            '',
            'coherent', 'coherence', 'coherency',
            'snoop', 'snoopy', 'snooping',
            'directory', 'snoop filter'
        ]
    ]

    words = []
    result = []
    for i in range(len(word_list[0])):
        words.append(word_list[0][i])
        for j in range(len(word_list[1])):
            words.append(word_list[1][j])
            for k in range(len(word_list[2])):
                words.append(word_list[2][k])
                result.append(' '.join(words))
                words.pop()
            words.pop()
        words.pop()

    for text in result:
        print(text.strip())
    
    return result


if __name__ == "__main__":
    combine_words()