import librosa
import music21 as m21
import pandas as pd
import os


'''
This function parses MusicXML file and extracts necessary score information as CSV.
'''
def readXmlAsCsv(xmlPath='xml/'):
    for subfolder in os.listdir(xmlPath):
        if subfolder.startswith('.'):
            continue
        subfolder_path = os.path.join(xmlPath, subfolder)
        for item in os.listdir(subfolder_path):
            if item.endswith('xml'):
                item_path = os.path.join(subfolder_path, item)
                xml_data = m21.converter.parse(item_path)
                print("Converting ", item_path)
                score = []
                for part in xml_data.parts:
                    for note in part.flat.notes:
                        if note.isChord:
                            print('note is chord: ', note)
                            measureNo = note.measureNumber
                            start = note.offset
                            duration = note.quarterLength

                            for chord_note in note:
                                pitch = chord_note.pitch
                                articulations = note.articulations
                                expressions = note.expressions
                                spanners = note.getSpannerSites()
                                gliss = []
                                for spanner in spanners:
                                    if 'Glissando' in spanner.classes:
                                        if spanner.isFirst(chord_note):
                                            gliss.append('slide start')
                                        if spanner.isLast(chord_note):
                                            gliss.append('slide last')
                                score.append(
                                    [measureNo, start, duration, pitch, m21.pitch.Pitch(pitch).frequency,
                                     articulations, expressions, gliss, spanners])

                        else:
                            measureNo = note.measureNumber
                            start = note.offset
                            duration = note.quarterLength
                            pitch = note.pitch
                            articulations = note.articulations
                            expressions = note.expressions
                            spanners = note.getSpannerSites()
                            gliss = []
                            for spanner in spanners:
                                if 'Glissando' in spanner.classes:
                                    if spanner.isFirst(note):
                                        gliss.append('slide start')
                                    if spanner.isLast(note):
                                        gliss.append('slide last')

                            score.append(
                                [measureNo, start, duration, pitch, m21.pitch.Pitch(pitch).frequency,
                                 articulations, expressions, gliss, spanners])
                score = sorted(score, key=lambda x: (x[0], x[1], x[2]))
                df = pd.DataFrame(score,
                                  columns=['MeasureNumber', 'Start', 'Duration', 'Pitch', 'f0',
                                           'Articulations', 'Expressions', 'Glissando', 'Spanner'])
                df.to_csv(os.path.join(path, 'csv', subfolder, os.path.splitext(item)[0] + '.csv'))