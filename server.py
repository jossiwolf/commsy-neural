import spacy
from flask import Flask, request, jsonify

app = Flask(__name__)

model_directory = "D:\\model"

nlp = spacy.load(model_directory)


@app.route('/getNormalizedClassLabel', methods=["POST"])
def getNormalizedName():
    labeled = nlp(request.form["originalName"])
    return jsonify(makeDictionaryFromLabels(labeled))


def makeDictionaryFromLabels(labeled):
    result = {}
    for ent in labeled.ents:
        result[str(ent.label_).lower()] = ent.text


if __name__ == '__main__':
    app.run()
