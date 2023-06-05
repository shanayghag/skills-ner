import re

def generateDataForNer(sentenceList, NerStartTag, NerEndTag, tagsBetweenSentenceList, textFilepath, fileSavepath):

    textFile = open(textFilepath, 'r')
    entities = textFile.readlines()

    generatedFile = open(fileSavepath, 'w')

    generatedSentenceCounter = 0

    for entity in entities:

        entity = re.sub('\n', '', entity)
        entity = re.sub(r'\[[0-9]+\]', '', entity)

        for sentence in tagsBetweenSentenceList:
            sentence = re.sub(r'entity', str(NerStartTag) + str(entity) + str(NerEndTag) + ' ', sentence)
            generatedFile.write(sentence)
            generatedSentenceCounter += 1

        for sentence in sentenceList:
            sentence = sentence + str(NerStartTag) + str(entity) + str(NerEndTag) + '\n'
            generatedFile.write(sentence)
            generatedSentenceCounter += 1

    generatedFile.close()
    textFile.close()

    print("Generated {} sentences!".format(generatedSentenceCounter))

if __name__ == '__main__':

    hobbySentences = ['I enjoy ', 'I like ', 'I am passionate about ', 'I am a professional in ', 'Love to watch ']
    musicHobbySentences = ['I enjoy ', 'I like ', 'I am passionate about ', 'I am a professional in ', 'Love to listen ']
    danceHobbySentences = ['I enjoy ', 'I like ', 'I am passionate about ', 'I am a professional in ', 'I have specialised in ', 'Would love to listen about ', 'I also enjoy talking about ']
    instrumentHobbySentences = ['I enjoy ', 'I also teach people to play ']
    HobbySentences = ['Along with my professional life, I also enjoy ', 'We can also talk about ', 'I also enjoy talking about ', 'Let us have chat about ', 'Would love to listen about ']
    professionalSentences = ['I am a ', 'Professional ', 'I am pursuing my career as ']

    generateDataForNer(sentenceList, NerStartTag, NerEndTag, tagsBetweenSentenceList, textFilepath, fileSavepath)

# print("Reading the file...")
# f = open("data/musical_instrument_hobbies.txt", 'r')
# lines = f.readlines()
#
# f1 = open("data/Instrument-NER.txt", "w")
#
# print("Done!!!")
#
# print("Generating data for NER...")
#
# for line in lines:
#     line = re.sub('\n', '', line)
#     line = re.sub(r'\[[0-9]+\]', '', line)
#
#     tempLine = "I am a <h>" + line + '</h> ' + 'player' + '\n'
#     Sentences.append(tempLine)
#     f1.write(tempLine)
#
#     tempLine = "Playing <h>" + line + '</h> ' + 'gives me happiness' + '\n'
#     Sentences.append(tempLine)
#     f1.write(tempLine)
#
#     for hobby in instrumentHobbySentences:
#         tempStr = hobby + '<h>' + line + '</h>\n'
#         f1.write(tempStr)
#         Sentences.append(tempStr)
#
# f.close()
#
# print("Number of sentences generated: {}".format(len(Sentences)))



