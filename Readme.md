# Commsy-Neural

Sooo... On Android, there's the wonderful `CollapsingToolbarLayout`. At least it's nice in theory.
As soon as you try to implement a custom title, well...

While building an app for Commsy (hamburg.schulcommsy.de), I encountered the problem of the names of rooms
containing lots of unnessecary information that made the title cut off. So I used spacy to train a NLP model
to extract all the nessecary information from the room names. This project serves as an API for it.

# Docs

Well. There's nothing really.

`/getNormalizedClassLabel`: `POST`

`parameters`: `originalName`. Name of the room to be shortened.