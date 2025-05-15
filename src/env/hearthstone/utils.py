# These functions are mainly inspired by Fireplace code, used as general utility functions

# Import Fireplace modules
from fireplace import cards 
from fireplace.deck import Deck
from fireplace.game import Game
from fireplace.player import Player
from hearthstone.enums import CardClass, CardType

# Import generic modules
import random
from xml.etree import ElementTree
from typing import List

def get_collection(card_class: CardClass) -> List[str]:
    """
    Return a list of collectible cards for the \a card_class
    
    :param card_class: CardClass
    :return: List of card ids
    """
    collection = []
    for card in cards.db.keys():
        cls = cards.db[card]
        if not cls.collectible:
            continue
        if cls.type == CardType.HERO:
            # Heroes are collectible...
            continue
        if cls.card_class and cls.card_class not in [card_class, CardClass.NEUTRAL]:
            # Play with more possibilities
            continue
        collection.append(cls.id)
    return collection

def random_draft(collection: List[str], include=[]) -> List[str]:
    """
    Return a deck of 30 random cards for the \a card_class
    
    :param card_class: CardClass
    :param exclude: List of card ids to exclude
    :param include: List of card ids to include
    :return: List of card ids
    """

    deck = list(include)

    while len(deck) < Deck.MAX_CARDS:
        card = random.choice(collection)
        if deck.count(card.id) < card.max_count_in_deck:
            deck.append(card.id)

    return deck


def random_class() -> CardClass:
    """
    Return a random CardClass
    
    :return: CardClass
    """
    return CardClass(random.randint(2, 10))

def setup_game(class1: int = None, class2: int = None, deck1: list = None, deck2: list = None) -> Game:
    """
    Setup a game with two random decks for player 1 and player 2
    If class1 or class2 is provided, the deck will be generated with the provided class
    
    :param class1: CardClass for player 1
    :param class2: CardClass for player 2
    :return: Game
    """
    # Make viable class list (2 to 10)
    class_list = list(range(2, 11))

    if not class1 or class1 not in class_list:
        card_class1 = random_class()
        print(f"Invalid class for player 1, using random class: {card_class1}")
    else:
        card_class1 = CardClass(class1)
    
    if not class2 or class2 not in class_list:
        card_class2 = random_class()
        print(f"Invalid class for player 2, using random class: {card_class2}")
    else:
        card_class2 = CardClass(class2)
        
    collection1 = get_collection(card_class1)
    collection2 = get_collection(card_class2)
        
    deck1 = random_draft(collection=collection1, include=deck1)
        
    deck2 = random_draft(collection=collection2, include=deck2)
        
    player1 = Player("Player1", deck1, card_class1.default_hero)
    player2 = Player("Player2", deck2, card_class2.default_hero)

    game = Game(players=(player1, player2))
    game.start()

    card_collections = [collection1, collection2]

    return game, card_collections
    
def entity_to_xml(entity) -> ElementTree.Element:
    """
    Convert a Fireplace Entity to an XML Element
    
    :param entity: Entity
    :return: ElementTree.Element
    """
    e = ElementTree.Element("Entity")
    for tag, value in entity.tags.items():
        if value and not isinstance(value, str):
            te = ElementTree.Element("Tag")
            te.attrib["enumID"] = str(int(tag))
            te.attrib["value"] = str(int(value))
            e.append(te)
    return e

