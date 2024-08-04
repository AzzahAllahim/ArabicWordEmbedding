model  = Word2Vec.load(model_name)
with io.open("questions-words.txt", "r",encoding='utf-8') as f:
        completed = 4
        pred_word_pure = []
        total_line = 0.0
        count = 0.0
        no_key_num = 0
        match = 0
        for line in f.readlines():
            v = line.strip().split(" ")
            if v[0] == ':':
                # if the line begins with ":", do nothing
                continue
            if (len(v) == 4):
                try:
                    pred_words = model.wv.most_similar(positive=[v[1], v[2]], negative=[v[0]], topn=5)
                    for word in pred_words:
                        pred_word_pure.append(simple_stem(word[0].encode('utf-8')))
                    if (len(pred_word_pure) == 0):
                        no_key_num += 1
                    # v[3] is the right answer
                    if any(v[3].encode('utf-8') in sl for sl in pred_word_pure):
                        count += 1
                    total_line += 1
                except KeyError:
                    no_key_num += 1
acc = count / total_line
print("The accuracy on the test data of ", model_name ," = %s" % acc)
