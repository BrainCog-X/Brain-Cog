from music21 import *
#s = converter.parse('../xmlfiles/four_part_hamony/ch4-03_A-major.xml')

s = corpus.parse('bach/bwv65.2.xml')
s.analyze('key')
print(len(s.parts))
s.show()

for i,part in enumerate(s.parts):
    print(i)
    print('-----------')
    for ns in part.flat.notes:
        print(ns.pitch)
        print(ns.duration.quarterLength)

note1 = note.Note("D5")
note2 = note.Note("F#5")
note2.duration.quarterLength = 0.5
note3 = note.Note("A5")

stream1 = stream.Stream()
stream1.append(note1)
stream1.append(note2)
stream1.append(note3)

print(note2.offset)

sout = stream1.getElementsByOffset(0,2)

sBach = corpus.parse('bach/bwv57.8')
s = sBach.chordify()
#cs = s.getElementsByClass('Chord')
s1 = s.flatten()
chords = s1.getElementsByClass('Chord')


# cMinor = chord.Chord(["A4","F4","D5"])
# print(cMinor.inversion())
# print(cMinor.isMinorTriad())

keyA = key.Key('B-')
for c in chords:
    rn = roman.romanNumeralFromChord(c, keyA)
    c.addLyric(str(rn.figure))

chords.show()

