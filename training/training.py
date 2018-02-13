import os
import random
from pathlib import Path

import plac
import spacy

MODEL_NAME = "de_commsy"
BASE_MODEL = "de"

# entity label
LABEL_KURS = "KURS"
LABEL_KLASSE = "KLASSE"

# load the german language model
nlp = spacy.load(BASE_MODEL)

TRAIN_DATA = [
    ("1718 Kurs S34 MGN Geographie", {
        'entities': [
            (18, 28, LABEL_KURS),
            (10, 17, LABEL_KLASSE)
        ]
    }),
    ("1718 Klasse 10d Physik", {
        'entities': [
            (16, 22, LABEL_KURS),
            (12, 15, LABEL_KLASSE)
        ]
    }),
    ("1718 Kurs S12 K Mathematik 2 Ba", {
        'entities': [
            (10, 26, LABEL_KURS),
            (10, 13, LABEL_KLASSE)
        ]
    }),
    ("1718 Kurs S12 K Mathematik 4 Ste", {
        'entities': [
            (10, 26, LABEL_KURS),
            (10, 13, LABEL_KLASSE)
        ]
    }),
    ("1718 Kurs 8 WP Medien", {
        'entities': [
            (15, 21, LABEL_KURS)
        ]
    }),
    ("1718 Klasse 9c Mathematik Frt", {
        'entities': [
            (15, 24, LABEL_KURS),
            (12, 14, LABEL_KLASSE)
        ]
    }),
    ("1718 Kurs 10 WP NaT", {
        'entities': [
            (16, 19, LABEL_KURS)
        ]
    }),
    ("1718 Kurs S12 Deutsch Hs", {
        'entities': [
            (14, 21, LABEL_KURS),
            (10, 13, LABEL_KLASSE)
        ]
    }),
    ("1718 Klasse 10d PGW", {
        'entities': [
            (16, 19, LABEL_KURS),
            (12, 15, LABEL_KLASSE)
        ]
    }),
    ("1718 Kurs 10 WP bili Natural Sciences", {
        'entities': [
            (21, 37, LABEL_KURS)
        ]
    }),
    ("1718 Kurs 9 WP Informatik", {
        'entities': [
            (15, 25, LABEL_KURS)
        ]
    }),
]


@plac.annotations(
    model=(BASE_MODEL, "option", "m", str),
    new_model_name=(MODEL_NAME, "option", "nm", str)
)
def main(model=BASE_MODEL, new_model_name=MODEL_NAME, n_iter=20):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank(BASE_MODEL)  # create blank Language class
        print("Created blank '" + BASE_MODEL + "' model")

    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe('ner')

    ner.add_label(LABEL_KURS)  # add new entity label to entity recognizer
    ner.add_label(LABEL_KLASSE)  # add new entity label to entity recognizer

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update([text], [annotations], sgd=optimizer, drop=0.35,
                           losses=losses)
            print(losses)

    # test the trained model
    test_text = [
        "1718 Klasse 9c Mathematik Frt",
        "1718 Kurs 10 WP bili Natural Sciences",
        "1718 Klasse 8a Chemie",
        "1718 Kurs S34 MGN Geographie",
        "1718 Kurs 9 WP Informatik",
        "1718 Kurs 10 WP Informatik"
    ]
    for value in test_text:
        doc = nlp(value)
        print("Entities in '%s'" % value)
        for ent in doc.ents:
            print(ent.label_, ent.text)

    output_dir = os.getcwd() + "\\model"

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text[0])
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == '__main__':
    plac.call(main)
