# Readme for adding decks to HearthGym
This readme file provides instructions on how to add new decks to HearthGym. It also provides information about the data structure used to store the decks and their metadata.

# Data Structure
## Deck Metadata
The metadata for decks is stored in the `src/data/decks_metadata.csv` file. This file contains the following columns:
- `deck_id`: A unique identifier for the deck.
- `hero_class`: The card class of the deck.
- `deck_name`: The name of the deck.
- `deck_description`: A description of the deck.

## Decks
The decks are stored in the `src/data/final_decks.csv` file. This file contains the following columns:
- `card_id`: The unique identifier for the card. (found in `src/data/cards_cleaned.csv`)
- `card_name`: The name of the card.
- `hero_class`: The card class of the deck.
- `deck_id`: The unique identifier for the deck. (found in `src/data/decks_metadata.csv`)

Adding new decks consists of two steps:
1. Adding the deck metadata to `src/data/decks_metadata.csv`.
2. Adding the cards to `src/data/final_decks.csv`.

# Adding New Decks to HearthGym
## Adding Deck Metadata
1. Open the `src/data/decks_metadata.csv` file.
2. Add a new row to the file with the following columns:
   - `deck_id`: A unique identifier for the deck. (e.g. `deck_1`)
   - `hero_class`: The card class of the deck. (e.g. `Druid`)
   - `deck_name`: The name of the deck. (e.g. `Aggro Druid`)
   - `deck_description`: A description of the deck. (e.g. `Aggro Druid is a fast-paced deck that focuses on dealing damage quickly.`)
3. Save the file.

## Adding Cards to Decks
1. Open the `src/data/final_decks.csv` file.    
2. Add a new row to the file with the following columns:
   - `card_id`: The unique identifier for the card. (found in `src/data/cards_cleaned.csv`)
   - `card_name`: The name of the card. (found in `src/data/cards_cleaned.csv`)
   - `hero_class`: The card class of the deck. (e.g. `Druid`)
   - `deck_id`: The unique identifier for the deck. (e.g. `deck_1`)
3. Save the file.
