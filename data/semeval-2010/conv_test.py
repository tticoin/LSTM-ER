import re, nltk, sys
from nltk.tokenize import StanfordTokenizer

tokenizer = StanfordTokenizer(r'../common/stanford-postagger-2015-04-20/stanford-postagger.jar')
for line in open("TEST_FILE.txt"):
    m = re.match(r'^([0-9]+)\s"(.+)"$', line.strip())
    if m is not None:
        txtfile = open("test/%s.txt" % m.group(1), 'w')
        annfile = open("test/%s.ann" % m.group(1), 'w')
        line = m.group(2)
        text = []
        t = line.split("<e1>")
        text.append(t[0])
        e1start = len(t[0])
        t = t[1].split("</e1>")
        e1 = t[0]
        text.append(t[0])
        e1end = len(t[0])+e1start
        t = t[1].split("<e2>")
        text.append(t[0])
        e2start = len(t[0])+e1end
        t = t[1].split("</e2>")
        text.append(t[0])
        e2 = t[0]
        e2end = len(t[0])+e2start
        text.append(t[1])
        text = " ".join(tokenizer.tokenize("".join(text)))
        txtfile.write(text)
        txtfile.write("\n")
        offset = 0
        err = False
        while e1 != text[e1start+offset:e1end+offset]:
            offset += 1
            if e1end+offset > len(text):
                break
        if e1end+offset > len(text):
            offset = 0
            e1 = " ".join(tokenizer.tokenize(e1))
            e1end = e1start + len(e1)
            while e1 != text[e1start+offset:e1end+offset]:
                offset += 1
                if e1end+offset > len(text):
                    print("%d\t%s" % (m.group(1), text))
                    err = True
                    break
        if not err:
            annfile.write("T1\tTerm %d %d\t%s\n" % (e1start+offset, e1end+offset, e1))
        err = False
        offset = 0
        while e2 != text[e2start+offset:e2end+offset]:
            offset+=1
            if e2end+offset > len(text):
                break
        if e2end+offset > len(text):
            offset = 0
            e2 = " ".join(tokenizer.tokenize(e2))
            e2end = e2start + len(e2)
            while e2 != text[e2start+offset:e2end+offset]:
                offset += 1
                if e2end+offset > len(text):
                    print("%d\t%s" % (m.group(1), text))
                    err = True
                    break
        if not err:
            annfile.write("T2\tTerm %d %d\t%s\n" % (e2start+offset, e2end+offset, e2))
        txtfile.close()
        annfile.close()
