import numpy as np

accurecies = []
model_count = 0 # to avoid writing the OOVW for each model with the same corpus!
f = open("WordsPairsFile.txt", "w")


    out_of_vocab = []
    accurecy = 0
    model = Word2Vec.load(model_name)
    #model = fasttext.load_model(model_name)
    total_distance = 0
    for x in range(0,len(words)):
    # Define two 1D arrays (vectors)
    # Form Fasttext models
        #word = model[words[x]]
        #syn = model[synonyms[x]]
    # for W2V models
      if words[x] in model.wv.vocab:
        word = model.wv[words[x]]
        if synonyms[x] in model.wv.vocab:
          syn = model.wv[synonyms[x]]
          dot_product = np.dot(word, syn)
    # Calculate the norms of the vectors
          norm_vector1 = norm(word)
          norm_vector2 = norm(syn)
          # Calculate the cosine similarity
          cosine_similarity = dot_product / (norm_vector1 * norm_vector2)
          # print("cosine_similarity:", cosine_similarity)
          total_distance = total_distance + cosine_similarity
        else:
            #f.write(synonyms[x].encode('utf-8') + '\n')
            if (model_count % 18 ==0):
              out_of_vocab.append(synonyms[x])
            continue
      else:
        #f.write(words[x].encode('utf-8') + '\n')
        if (model_count % 18 == 0):
           out_of_vocab.append(words[x])
        continue
    #Final accurecy for the model for all pairs in the dataset
    accurecy = total_distance/len(words)
    print("Model ", model_name," Done! # of OOV: ", len(out_of_vocab))
