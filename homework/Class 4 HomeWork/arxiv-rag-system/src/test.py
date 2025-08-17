import more_itertools

def main():

    text = 'If you try out a few side projects using RAG or agents, you can also point to these when interviewing and say you have an interest in the field and some skills in the relevant frameworks'

    chunk_size = 50
    step = 2

    print('Fixed chunking:\n')
    for chunk in more_itertools.chunked(text, chunk_size):
        print(''.join(chunk))

    print('\nSliding window:\n')

    # Sliding window
    for chunk in more_itertools.windowed(text, chunk_size, fillvalue='', step=step):
        print(''.join(chunk))


if __name__=="__main__":
    main()
    

