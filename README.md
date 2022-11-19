# IMDB_movie_classification

Project contains experiments on movie's genre prediction task.

Task: predict movie's genre, based on known title, plot and poster of the movie.

Different approaches for text and image feature extraction were used in combination of few classification method.

The best combination: image features from `resnet50` model, text features from `bag of words` method and `Logistic Regression` as a classification method with mean `F1 score` `0.675` (among all genres).

Main genre classification problem - treating `Superhero` as class (genre), meanwhile in IMDB base it's only marked as tag.
