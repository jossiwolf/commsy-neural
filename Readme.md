# Commsy-Neural

Sooo... On Android, there's the wonderful `CollapsingToolbarLayout`. At least it's nice in theory.
As soon as you try to implement a custom title, well...

While building an app for CommSy (hamburg.schulcommsy.de), I encountered the problem of the names of rooms
containing lots of unnecessary information that made the title cut off. So I used spaCy to train a NLP model
to extract all the necessary information from the room names. This project serves as an API for it.

This was hacked together from the spaCy samples in about half an hour, so the code may not be the prettiest (yet!).

# Docs

Well. There's nothing much really.

`/getNormalizedClassLabel`: `POST`

`parameters`: `originalName`. Name of the room to be shortened.

# PRs/Issues

PRs/Issues are always welcome!