#Initialising the game info and state variables for Balatro

game_info = {
    "jokers": [i for i in range(0, 150)],
    "deck_cards": [i for i in range(0, 52)],
    "tarot_cards": [i for i in range(0, 22)],
    "spectral_cards": [i for i in range(0, 18)],
    "planet_cards": [i for i in range(0, 12)],
    "vouchers": [i for i in range(0, 16)],
    "booster_packs": [i for i in range(0, 34)],
    "card_modifier": [i for i in range(0, 9)],
    "seal": [0 for i in range(0, 4)],
    "edition": [0 for i in range(0, 5)],
    
}
current_state = {
    "owned_jokers": {
        "joker_1": {"id": 0, "seal": 0, "edition": 0, "card_modifier": 0},
        "joker_2": {"id": 0, "seal": 0, "edition": 0, "card_modifier": 0},
        "joker_3": {"id": 0, "seal": 0, "edition": 0, "card_modifier": 0},
        "joker_4": {"id": 0, "seal": 0, "edition": 0, "card_modifier": 0},
        "joker_5": {"id": 0, "seal": 0, "edition": 0, "card_modifier": 0},
    },
    "owned_deck_cards": [game_info["deck_cards"]],
    "money": 0,
    "consumables": [],
    "owned_vouchers": [],
    "available_shop_items": [],
    "planet_levels": [1 for i in range(0, 12)],
    "ante": 1,
    "round": 1,
    "hands_played": 0,
    "scored_chips": 0,
    "hand_played": False,
    "hands": 0,
    "discards": 0,
    "cards_seleced": 0,
}
